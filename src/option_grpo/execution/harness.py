"""Execution harness that runs inside the sandbox.

This file is deliberately self-contained (standard library only): it is copied into the
sandbox working directory and executed there as ``python harness.py job.json out.json``.

It executes a candidate program in stages that mirror the layered execution reward:

    parse -> imports -> setup -> run -> output -> expected type -> property tests

and reports the furthest stage reached, the observable output (type, size, digest), the
SDK calls the program made (observed with a profiler hook, so no SDK code is patched)
and the outcome of every property test.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import hashlib
import importlib
import io
import json
import os
import re
import sys
import time
import traceback

SOLUTION_FILE = "<solution>"
SETUP_FILE = "<setup>"

STAGES = (
    "parse_error",  # the program is not valid Python
    "import_error",  # an import of the program failed
    "setup_error",  # the task's own setup failed (an infrastructure problem)
    "runtime_error",  # the program raised
    "output_none",  # the program ran but produced no value
    "type_mismatch",  # the program ran and produced a value of an unexpected type
    "type_ok",  # the output has the expected type
    "tests_passed",  # all property tests pass
)


# --------------------------------------------------------------------------------------
# Resource limits
# --------------------------------------------------------------------------------------
def apply_limits(limits):
    memory_mb = limits.get("memory_mb")
    if memory_mb and sys.platform.startswith("linux"):
        try:
            import resource

            limit = int(memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except Exception:
            pass
    if not limits.get("allow_network", False):
        import socket

        class _NoNetworkSocket(socket.socket):
            def connect(self, address):
                if self.family == getattr(socket, "AF_UNIX", None):
                    return super().connect(address)
                raise PermissionError("network access is disabled in the sandbox")

            def connect_ex(self, address):
                raise PermissionError("network access is disabled in the sandbox")

        socket.socket = _NoNetworkSocket


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------
def write_fixtures(fixtures):
    for item in fixtures or []:
        path = item["path"]
        if item["type"] == "dir":
            os.makedirs(path, exist_ok=True)
            continue
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if item["type"] == "bytes":
            with open(path, "wb") as handle:
                handle.write(base64.b64decode(item["b64"]))
        else:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(item["content"])


# --------------------------------------------------------------------------------------
# Observable output
# --------------------------------------------------------------------------------------
_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")
# Zero-argument methods that expose the content of a value; lazy objects are materialised
# (``collect``/``compute``) so that equality is decided on data, not on object identity.
_EXPORTERS = (
    "to_pydict", "to_pylist", "to_dict", "to_list", "tolist", "model_dump", "as_dict", "dict", "to_json",
    "collect", "compute", "to_arrow", "to_pandas",
)


def canonical(value, depth=0):
    """JSON-serialisable canonical form of a value, used for output equality."""
    if depth > 6:
        return {"__repr__": _ADDRESS.sub("0x", repr(type(value)))}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return round(value, 9) if value == value else "nan"
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes__": hashlib.sha256(bytes(value)).hexdigest()}
    if isinstance(value, (list, tuple)):
        return [canonical(v, depth + 1) for v in value[:10000]]
    if isinstance(value, (set, frozenset)):
        return sorted(json.dumps(canonical(v, depth + 1), sort_keys=True, default=str) for v in value)
    if isinstance(value, dict):
        items = [(str(k), canonical(v, depth + 1)) for k, v in list(value.items())[:10000]]
        return {"__dict__": sorted(items, key=lambda kv: kv[0])}
    for name in _EXPORTERS:
        method = getattr(value, name, None)
        if callable(method):
            try:
                exported = method()
            except Exception:
                continue
            if exported is not value:
                return {"__type__": type(value).__name__, "value": canonical(exported, depth + 1)}
    state = getattr(value, "__dict__", None)
    if isinstance(state, dict) and state and depth < 3:
        public = {k: v for k, v in state.items() if not callable(v)}
        return {"__type__": type(value).__name__, "state": canonical(public, depth + 1)}
    return {"__repr__": _ADDRESS.sub("0x", repr(value))[:2000]}


def digest(value):
    try:
        payload = json.dumps(canonical(value), sort_keys=True, default=str)
    except Exception:
        payload = "unserialisable:" + type(value).__name__
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:24]


def size_of(value):
    """Length or leading dimension of a value, or None when it has neither."""
    try:
        if hasattr(type(value), "__len__"):
            return int(len(value))
        shape = getattr(value, "shape", None)
        if isinstance(shape, (tuple, list)) and shape and isinstance(shape[0], int):
            return int(shape[0])
        rows = getattr(value, "num_rows", None)
        if isinstance(rows, int):
            return rows
    except Exception:
        return None
    return None


def type_names(value):
    names = []
    for cls in type(value).__mro__:
        names.append(cls.__module__ + "." + cls.__qualname__)
        names.append(cls.__qualname__)
    return names


def accepts(spec, value):
    """Does ``value`` satisfy a runtime acceptance spec (see ``runtime_accept``)?"""
    if value is None:
        return bool(spec.get("none_ok"))
    if spec.get("any"):
        return True
    wanted = set(spec.get("names") or [])
    if wanted and wanted.intersection(type_names(value)):
        return True
    for protocol in spec.get("structural") or []:
        cls = type(value)
        if protocol == "sequence" and hasattr(cls, "__len__") and hasattr(cls, "__getitem__"):
            return True
        if protocol == "sized" and hasattr(cls, "__len__"):
            return True
        if protocol == "iterable" and hasattr(cls, "__iter__"):
            return True
        if protocol == "async_iterable" and hasattr(cls, "__aiter__"):
            return True
        if protocol == "mapping" and hasattr(cls, "keys") and hasattr(cls, "__getitem__"):
            return True
        if protocol == "callable" and callable(value):
            return True
    return False


# --------------------------------------------------------------------------------------
# Call tracing
# --------------------------------------------------------------------------------------
class Tracer:
    """Records the SDK calls made directly by the program (profiler based)."""

    def __init__(self, roots, functions, invariants):
        self.roots = tuple(roots)
        self.root_dirs = []
        self.code_to_id = {}
        self.object_to_id = {}
        self.invariants = invariants  # function id -> [(expr, compiled code)]
        self.calls = []
        self._frames = {}
        self._resolve(functions)

    def _resolve(self, functions):
        for root in self.roots:
            module = sys.modules.get(root)
            location = getattr(module, "__file__", None)
            if location:
                base = os.path.dirname(os.path.abspath(location))
                self.root_dirs.append(base if os.path.basename(location).startswith("__init__") else os.path.abspath(location))
        ambiguous = set()
        for entry in functions or []:
            try:
                target = importlib.import_module(entry["module"])
                for attr in entry["attrs"]:
                    target = target.__dict__[attr] if isinstance(target, type) and attr in target.__dict__ else getattr(target, attr)
            except Exception:
                continue
            seen = set()
            stack = [target]
            while stack:
                obj = stack.pop()
                if id(obj) in seen or obj is None:
                    continue
                seen.add(id(obj))
                self.object_to_id[id(obj)] = (obj, entry["id"])
                code = getattr(obj, "__code__", None)
                if code is not None:
                    if code in self.code_to_id and self.code_to_id[code] != entry["id"]:
                        ambiguous.add(code)
                    self.code_to_id[code] = entry["id"]
                for link in ("__wrapped__", "__func__", "fget", "func"):
                    stack.append(getattr(obj, link, None))
        for code in ambiguous:  # a decorator wrapper shared by several functions
            del self.code_to_id[code]

    def _is_sdk_file(self, filename):
        return any(filename.startswith(base) for base in self.root_dirs)

    def _identify(self, frame):
        code = frame.f_code
        fid = self.code_to_id.get(code)
        if fid is not None:
            return fid
        for value in frame.f_locals.values():  # decorator wrappers keep the function in a closure
            hit = self.object_to_id.get(id(value))
            if hit is not None and hit[0] is value:
                return hit[1]
        return None

    def __call__(self, frame, event, arg):
        if event == "call":
            caller = frame.f_back
            if caller is None or caller.f_code.co_filename != SOLUTION_FILE:
                return
            code = frame.f_code
            fid = self.code_to_id.get(code)
            if fid is None and not self._is_sdk_file(code.co_filename):
                return
            known = self._frames.get(id(frame))
            if known is not None and known[0] is frame:
                return  # a generator or coroutine being resumed
            if fid is None:
                fid = self._identify(frame)
            module = frame.f_globals.get("__name__", "")
            qualname = getattr(code, "co_qualname", code.co_name)
            entry = {"id": fid, "name": module + "." + qualname, "in_size": None, "ret_size": None, "ret_types": None, "invariants": []}
            arguments = dict(frame.f_locals)
            for value in arguments.values():
                measured = size_of(value)
                if measured is not None and not isinstance(value, (str, bytes)):
                    entry["in_size"] = measured
                    break
            for expr, compiled in self.invariants.get(fid, ()):
                try:
                    entry["invariants"].append([expr, bool(eval(compiled, {"__builtins__": __builtins__}, arguments))])
                except Exception:
                    entry["invariants"].append([expr, None])
            self._frames[id(frame)] = (frame, entry)
            self.calls.append(entry)
        elif event == "return":
            known = self._frames.get(id(frame))
            if known is not None and known[0] is frame:
                entry = known[1]
                if arg is not None:
                    entry["ret_types"] = type_names(arg)
                    entry["ret_size"] = size_of(arg)
                    entry["_ret"] = arg
        elif event == "c_call":
            if frame.f_code.co_filename != SOLUTION_FILE:
                return
            module = getattr(arg, "__module__", None)
            owner = getattr(arg, "__self__", None)
            if not module and owner is not None:
                module = getattr(type(owner), "__module__", None) if not isinstance(owner, type) else owner.__module__
            if module and module.split(".")[0] in self.roots:
                self.calls.append({"id": None, "name": module + "." + getattr(arg, "__qualname__", "?"), "in_size": None,
                                   "ret_size": None, "ret_types": None, "invariants": [], "native": True})

    def export(self):
        out = []
        for entry in self.calls:
            item = {k: v for k, v in entry.items() if not k.startswith("_") and k != "ret_types"}
            names = entry.get("ret_types")
            item["ret_type"] = names[0] if names else None
            out.append(item)
        return out


# --------------------------------------------------------------------------------------
# One execution of setup + program
# --------------------------------------------------------------------------------------
class Run:
    def __init__(self):
        self.stage = None
        self.error = None
        self.output = None
        self.has_output = False
        self.tracer = None


def split_program(tree):
    """Separate top-level imports from the body and find how to read the output."""
    imports, body = [], []
    for stmt in tree.body:
        (imports if isinstance(stmt, (ast.Import, ast.ImportFrom)) else body).append(stmt)
    output_name = None
    if body:
        last = body[-1]
        if isinstance(last, ast.Expr):
            assign = ast.Assign(targets=[ast.Name(id="__output__", ctx=ast.Store())], value=last.value)
            body[-1] = ast.copy_location(assign, last)
            output_name = "__output__"
        elif isinstance(last, ast.Assign) and isinstance(last.targets[0], ast.Name):
            output_name = last.targets[0].id
        elif isinstance(last, (ast.AnnAssign, ast.AugAssign)) and isinstance(last.target, ast.Name):
            output_name = last.target.id
    import_module = ast.Module(body=imports, type_ignores=[])
    body_module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(import_module)
    ast.fix_missing_locations(body_module)
    return import_module, body_module, output_name


def describe_error(exc):
    tb = traceback.extract_tb(exc.__traceback__)
    line = next((frame.lineno for frame in reversed(tb) if frame.filename == SOLUTION_FILE), None)
    return {"type": type(exc).__name__, "message": str(exc)[:500], "line": line}


def execute(job, setup_code, fixtures, trace_spec):
    """Run setup and program once in a fresh namespace."""
    run = Run()
    try:
        tree = ast.parse(job["solution"], filename=SOLUTION_FILE)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        run.stage, run.error = "parse_error", {"type": type(exc).__name__, "message": str(exc)[:500], "line": getattr(exc, "lineno", None)}
        return run
    import_module, body_module, output_name = split_program(tree)
    namespace = {"__name__": "__solution__", "__builtins__": __builtins__}
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            write_fixtures(fixtures)
            exec(compile(job.get("prelude", ""), SETUP_FILE, "exec"), namespace)
        except BaseException as exc:  # the SDK itself cannot be imported
            run.stage, run.error = "import_error", describe_error(exc)
            return run
        try:
            exec(compile(setup_code or "", SETUP_FILE, "exec"), namespace)
        except BaseException as exc:
            run.stage, run.error = "setup_error", describe_error(exc)
            return run
        try:
            exec(compile(import_module, SOLUTION_FILE, "exec"), namespace)
        except BaseException as exc:
            run.stage, run.error = "import_error", describe_error(exc)
            return run
        tracer = Tracer(job.get("sdk_roots", []), trace_spec["functions"], trace_spec["invariants"])
        run.tracer = tracer
        code = compile(body_module, SOLUTION_FILE, "exec")
        sys.setprofile(tracer)
        try:
            exec(code, namespace)
        except BaseException as exc:
            sys.setprofile(None)
            if isinstance(exc, (KeyboardInterrupt,)):
                raise
            run.stage, run.error = "runtime_error", describe_error(exc)
            return run
        finally:
            sys.setprofile(None)
    if output_name is not None and output_name in namespace:
        run.output, run.has_output = namespace[output_name], True
    elif "result" in namespace:
        run.output, run.has_output = namespace["result"], True
    run.stage = "executed"
    return run


# --------------------------------------------------------------------------------------
# Property tests
# --------------------------------------------------------------------------------------
def _calls_of(run, function_id):
    return [c for c in run.tracer.calls if c.get("id") == function_id] if run.tracer else []


def evaluate_tests(job, base, trace_spec):
    results = []
    rerun = None
    for test in job.get("tests", []):
        kind = test["kind"]
        outcome = {"id": test.get("id", kind), "kind": kind, "passed": False, "skipped": False, "detail": ""}
        try:
            if kind == "type":
                outcome["passed"] = accepts(test["accept"], base.output)
                if not outcome["passed"]:
                    outcome["detail"] = "output type " + type(base.output).__name__
            elif kind == "output_shape":
                measured = size_of(base.output)
                outcome["passed"] = measured is not None and measured >= 0
                outcome["skipped"] = measured is None
                outcome["passed"] = outcome["passed"] or outcome["skipped"]
            elif kind == "call_type":
                calls = [c for c in _calls_of(base, test["function"]) if "_ret" in c]
                if not calls:
                    outcome["skipped"], outcome["passed"] = True, True
                else:
                    bad = [c for c in calls if not accepts(test["accept"], c["_ret"])]
                    outcome["passed"] = not bad
                    if bad:
                        outcome["detail"] = "returned " + type(bad[0]["_ret"]).__name__
            elif kind == "shape":
                calls = [c for c in _calls_of(base, test["function"]) if c.get("ret_size") is not None]
                relation = test.get("relation", "defined")
                checked = [c for c in calls if relation == "defined" or c.get("in_size") is not None]
                if not checked:
                    outcome["skipped"], outcome["passed"] = True, True
                else:
                    ok = True
                    for call in checked:
                        if call["ret_size"] < 0:
                            ok = False
                        elif relation == "<=" and not call["ret_size"] <= call["in_size"]:
                            ok = False
                        elif relation == "==" and call["ret_size"] != call["in_size"]:
                            ok = False
                    outcome["passed"] = ok
            elif kind == "invariant":
                checks = [flag for c in _calls_of(base, test["function"]) for expr, flag in c["invariants"] if expr == test["expr"]]
                checks = [flag for flag in checks if flag is not None]
                if not checks:
                    outcome["skipped"], outcome["passed"] = True, True
                else:
                    outcome["passed"] = all(checks)
            elif kind == "determinism":
                if rerun is None:
                    rerun = execute(job, job.get("setup", ""), job.get("fixtures"), trace_spec)
                same_stage = rerun.stage == "executed"
                outcome["passed"] = same_stage and digest(rerun.output) == digest(base.output)
                if not outcome["passed"]:
                    outcome["detail"] = "second run differs" if same_stage else "second run failed: " + str(rerun.stage)
            elif kind == "edge":
                variant = next((v for v in job.get("variants", []) if v["name"] == test["variant"]), None)
                if variant is None:
                    outcome["skipped"], outcome["passed"] = True, True
                else:
                    run = execute(job, variant["setup"], variant.get("fixtures"), trace_spec)
                    expect = test.get("expect", "any")
                    if run.stage == "setup_error":
                        outcome["skipped"], outcome["passed"] = True, True
                    elif run.stage == "executed":
                        type_ok = accepts(job.get("expected", {"any": True}), run.output)
                        outcome["passed"] = type_ok and expect in ("any", "ok")
                        if isinstance(expect, dict) and "digest" in expect:
                            outcome["passed"] = type_ok and digest(run.output) == expect["digest"]
                        if not outcome["passed"]:
                            outcome["detail"] = "unexpected result on variant " + test["variant"]
                    elif run.stage == "runtime_error":
                        raised = run.error["type"]
                        if isinstance(expect, dict) and "raises" in expect:
                            outcome["passed"] = raised == expect["raises"]
                        else:
                            # Without a validated expectation a well-typed domain error is accepted,
                            # programming errors are not.
                            outcome["passed"] = expect == "any" and raised not in (
                                "NameError", "AttributeError", "TypeError", "SyntaxError", "IndexError",
                                "KeyError", "UnboundLocalError", "ZeroDivisionError", "RecursionError", "ImportError",
                                "ModuleNotFoundError",
                            )
                        if not outcome["passed"]:
                            outcome["detail"] = "raised " + raised + " on variant " + test["variant"]
                    else:
                        outcome["detail"] = "stage " + str(run.stage)
            elif kind == "reference":
                outcome["passed"] = digest(base.output) == test["digest"]
                if not outcome["passed"]:
                    outcome["detail"] = "output differs from the reference"
            else:
                outcome["skipped"], outcome["passed"] = True, True
                outcome["detail"] = "unknown test kind"
        except Exception as exc:  # a failing check must never crash the harness
            outcome["passed"] = False
            outcome["detail"] = "check failed: " + type(exc).__name__ + ": " + str(exc)[:200]
        if outcome["skipped"] and test.get("required"):
            # The property belongs to a function the program never called, so it was not
            # demonstrated; a required property cannot be satisfied vacuously.
            outcome["passed"] = False
            outcome["detail"] = "the function under test was not called"
        results.append(outcome)
    return results


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def run_job(job):
    started = time.time()
    for extra in job.get("sys_path", []):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    invariants = {}
    for test in job.get("tests", []):
        if test.get("kind") == "invariant":
            try:
                compiled = compile(test["expr"], "<invariant>", "eval")
            except SyntaxError:
                continue
            invariants.setdefault(test["function"], []).append((test["expr"], compiled))
    trace_spec = {"functions": job.get("functions", []), "invariants": invariants}
    base = execute(job, job.get("setup", ""), job.get("fixtures"), trace_spec)
    result = {"stage": base.stage, "error": base.error, "output": None, "trace": [], "tests": [], "n_api": 0}
    if base.stage == "executed":
        expected = job.get("expected") or {"any": True}
        output = base.output
        result["output"] = {
            "type": type(output).__module__ + "." + type(output).__qualname__,
            "digest": digest(output),
            "size": size_of(output),
            "repr": _ADDRESS.sub("0x", repr(output))[:300],
            "is_none": output is None,
        }
        result["n_api"] = len(base.tracer.calls)
        if output is None and not expected.get("none_ok"):
            result["stage"] = "output_none"
        elif not accepts(expected, output):
            result["stage"] = "type_mismatch"
        else:
            result["stage"] = "type_ok"
            tests = evaluate_tests(job, base, trace_spec)
            result["tests"] = tests
            if all(t["passed"] for t in tests):
                result["stage"] = "tests_passed"
        result["trace"] = base.tracer.export()
    result["duration_s"] = round(time.time() - started, 4)
    return result


def main(argv):
    job_path, out_path = argv[1], argv[2]
    with open(job_path, encoding="utf-8") as handle:
        job = json.load(handle)
    apply_limits(job.get("limits", {}))
    try:
        result = run_job(job)
    except BaseException as exc:  # includes SystemExit raised by the program
        result = {"stage": "runtime_error", "error": {"type": type(exc).__name__, "message": str(exc)[:500], "line": None},
                  "output": None, "trace": [], "tests": [], "n_api": 0, "duration_s": 0.0}
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(result, handle, default=str)
    os.replace(tmp, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
