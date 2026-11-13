"""Curriculum task generation.

The task distribution ``D_k`` of curriculum level ``k`` is generated from paths of the
SDK graphs:

* ``D_0`` -- one task per public function (a single API call);
* ``D_1`` -- composable paths of 2..``L_max`` calls sampled from ``G_flow``;
* ``D_2`` -- full programs: longer walks through ``G_flow``.

For every path a call skeleton is synthesised from type annotations.  It provides the
setup code shown in the prompt, the minimal number of API calls ``N_min`` and the
reference length ``T_ref``; the skeleton itself is never shown to the policy.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any

from option_grpo.config import BenchmarkConfig, DiscoveryConfig
from option_grpo.graphs.build import SDKGraphs
from option_grpo.options.discovery import enumerate_paths
from option_grpo.proptest.generator import function_locator, runtime_accept, task_tests
from option_grpo.sdk.introspect import FunctionInfo
from option_grpo.sdk.types import TypeRef
from option_grpo.tasks.dataset import Task, TaskSet
from option_grpo.tasks.fixtures import Fixture
from option_grpo.tasks.synth import SynthError, Synthesizer, describe_function
from option_grpo.utils import get_logger, stable_hash, stable_unit_float

log = get_logger(__name__)

_VARIANT_ORDER = ("empty", "single", "boundary")


def _snake(name: str) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    return text or "value"


@dataclass
class _Program:
    """A synthesised reference program under construction."""

    setup_lines: list[str] = field(default_factory=list)
    setup_types: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    fixtures: dict[str, Fixture] = field(default_factory=dict)
    imports: set[str] = field(default_factory=set)
    variant_exprs: dict[str, dict[str, str]] = field(default_factory=dict)  # variant -> var -> source
    setup_exprs: dict[str, str] = field(default_factory=dict)  # var -> source
    statements: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)
    n_calls: int = 0
    used: set[str] = field(default_factory=lambda: {"result"})

    def new_var(self, base: str) -> str:
        base = _snake(base)
        if base in {"self", "cls", "class", "type", "input", "id", "list", "dict", "str", "int", "filter", "map"}:
            base = f"{base}_value"
        name = base
        suffix = 2
        while name in self.used:
            name = f"{base}_{suffix}"
            suffix += 1
        self.used.add(name)
        return name

    def add_setup(self, var: str, code: str, vtype: TypeRef | None, variants: dict[str, str] | None = None) -> None:
        self.setup_lines.append(f"{var} = {code}")
        self.setup_exprs[var] = code
        self.setup_types[var] = vtype.to_json() if vtype is not None else None
        for name, alt in (variants or {}).items():
            self.variant_exprs.setdefault(name, {})[var] = alt

    def absorb(self, fixtures: list[Fixture], imports: set[str]) -> None:
        for fixture in fixtures:
            self.fixtures.setdefault(fixture.path, fixture)
        self.imports |= imports


def _format_chain(target: str, segments: list[str], width: int = 88) -> str:
    """``target = a.b().c()`` on one line, or as a parenthesised method chain."""
    flat = f"{target} = {''.join(segments)}"
    if len(flat) <= width or len(segments) < 3:
        return flat
    body = "\n".join(f"    {segment}" for segment in segments)
    return f"{target} = (\n{body}\n)"


class TaskGenerator:
    """Generates the curriculum tasks of one SDK."""

    def __init__(
        self,
        graphs: SDKGraphs,
        prelude: str,
        cfg: BenchmarkConfig | None = None,
        discovery: DiscoveryConfig | None = None,
        hints: dict[str, str] | None = None,
        min_compat: float = 0.5,
    ) -> None:
        self.graphs = graphs
        self.index = graphs.index
        self.prelude = prelude
        self.cfg = cfg or BenchmarkConfig()
        self.discovery = discovery or DiscoveryConfig()
        self.synth = Synthesizer(graphs, hints, min_compat)
        self.skipped: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ reference
    def _via(self, previous: FunctionInfo, current: FunctionInfo) -> str:
        data = self.graphs.flow.get_edge_data(previous.id, current.id)
        if not data or not data.get("via"):
            raise SynthError(f"no flow edge {previous.id} -> {current.id}")
        vias = data["via"]
        return "self" if "self" in vias else vias[0]

    def build_program(self, functions: list[FunctionInfo], level: int) -> _Program:
        """Synthesise the reference skeleton for a path."""
        synth = self.synth
        prog = _Program()
        chain: list[str] | None = None  # source segments of the current flow value
        for step, fn in enumerate(functions):
            supplied: dict[str, str] = {}
            receiver: str | None = None
            head: list[str] | None = None
            if step > 0:
                assert chain is not None
                via = self._via(functions[step - 1], fn)
                if via == "self":
                    head = chain
                else:
                    if len(chain) > 1 or not chain[0].isidentifier():
                        var = prog.new_var(_snake(functions[step - 1].returns.name) if functions[step - 1].returns else f"step_{step}")
                        prog.statements.append(_format_chain(var, chain))
                        chain = [var]
                    supplied[via] = chain[0]
            if fn.receiver is not None and head is None:
                built = synth.build(fn.receiver)
                var = prog.new_var(fn.receiver.name)
                prog.add_setup(var, built.code, fn.receiver)
                prog.absorb(built.fixtures, built.imports)
                receiver = var
                head = [var]
            position = 0
            for param in fn.params:
                if param.name in supplied or param.kind == "kwarg":
                    continue
                if not param.required and param.kind != "vararg":
                    continue
                if param.kind == "vararg" and param.type is None:
                    continue
                if synth.is_object_type(param.type):
                    value = synth.argument(fn, param, position, 0, (fn.id,))
                    prog.absorb(value.fixtures, value.imports)
                    if step == 0 or level == 0:
                        var = prog.new_var(param.name)
                        prog.add_setup(var, value.code, param.type)
                        supplied[param.name] = var
                    else:
                        supplied[param.name] = value.code
                        prog.n_calls += value.n_calls
                        prog.details.append(f"`{param.name}` of step {step + 1}: {value.description}")
                else:
                    value = synth.primitive(fn, param, position)
                    position += 1
                    prog.absorb(value.fixtures, value.imports)
                    var = prog.new_var(param.name)
                    prog.add_setup(var, value.code, value.type, value.variants)
                    supplied[param.name] = var
            if fn.is_async:
                prog.imports.add("import asyncio")
            if fn.kind in {"method", "property"}:
                assert head is not None
                call = synth.render_call(fn, "", supplied)  # ".name(args)"
                if fn.is_async:
                    inner = "".join(head) + call[len("asyncio.run("):-1]
                    chain = [f"asyncio.run({inner})"]
                else:
                    chain = head + [call]
            else:
                chain = [synth.render_call(fn, receiver, supplied)]
            if fn.kind != "property":
                prog.n_calls += 1
        assert chain is not None
        prog.statements.append(_format_chain("result", chain))
        return prog

    # ------------------------------------------------------------------ description
    def describe(self, functions: list[FunctionInfo], prog: _Program) -> str:
        phrases = [describe_function(fn) for fn in functions]
        if len(phrases) == 1:
            sentence = phrases[0]
        else:
            sentence = ", then ".join(phrases)
        sentence = sentence[0].upper() + sentence[1:] + "."
        lines = [sentence]
        if prog.setup_exprs:
            names = ", ".join(f"`{name}`" for name in prog.setup_exprs)
            lines.append(f"Use the values defined in the setup ({names}).")
        lines.extend(f"- {detail}." for detail in prog.details)
        lines.append("Store the final value in `result`.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ variants
    def _variants(self, prog: _Program) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        data_fixtures = [f for f in prog.fixtures.values() if f.kind not in {"dir", "png", "pdf"}]
        for name in _VARIANT_ORDER:
            exprs = prog.variant_exprs.get(name, {})
            swap_files = name in {"empty", "single"} and bool(data_fixtures)
            if not exprs and not swap_files:
                continue
            setup = "\n".join(f"{var} = {exprs.get(var, code)}" for var, code in prog.setup_exprs.items())
            fixtures = [
                (f.with_variant(name) if swap_files and f in data_fixtures else f).to_json()
                for f in prog.fixtures.values()
            ]
            out.append({"name": name, "setup": setup, "fixtures": fixtures})
        return out

    # ------------------------------------------------------------------ assembly
    def build_task(self, functions: list[FunctionInfo], level: int) -> Task:
        prog = self.build_program(functions, level)
        variants = self._variants(prog)
        prelude = "\n".join([self.prelude.rstrip(), *sorted(prog.imports)])
        path = [fn.id for fn in functions]
        task_id = f"{self.index.name}-L{level}-{stable_hash(level, *path, digits=10)}"
        split = "test" if stable_unit_float("split", task_id) < self.cfg.test_fraction else "train"
        return Task(
            id=task_id,
            sdk=self.index.name,
            level=level,
            path=path,
            call_names=[fn.name for fn in functions],
            description=self.describe(functions, prog),
            prelude=prelude,
            setup="\n".join(prog.setup_lines),
            fixtures=[f.to_json() for f in prog.fixtures.values()],
            setup_types=prog.setup_types,
            expected=runtime_accept(functions[-1].returns, self.index),
            tests=task_tests(functions, self.index, [v["name"] for v in variants], self.cfg.max_tests_per_function),
            variants=variants,
            reference="\n".join(prog.statements) + "\n",
            n_min=prog.n_calls,
            split=split,
            meta={"functions": [function_locator(fn) for fn in functions]},
        )

    def _try(self, path: tuple[str, ...], level: int) -> Task | None:
        functions = [self.index.functions[fid] for fid in path]
        try:
            return self.build_task(functions, level)
        except SynthError as exc:
            self.skipped.append({"level": level, "path": list(path), "reason": str(exc)})
            return None

    # ------------------------------------------------------------------ levels
    def level0(self) -> list[Task]:
        tasks = []
        for fn in self.graphs.functions:
            task = self._try((fn.id,), 0)
            if task is not None:
                tasks.append(task)
        return tasks

    def level1(self) -> list[Task]:
        target = max(int(round(self.cfg.level1_per_function * self.graphs.flow.number_of_nodes())), 1)
        rng = random.Random(stable_hash("level1", self.index.name, self.cfg.seed))
        by_start: dict[str, list[tuple[tuple[str, ...], float]]] = {}
        for path, weight in enumerate_paths(self.graphs, self.discovery):
            by_start.setdefault(path[0], []).append((path, weight))
        for start in by_start:
            # Weighted shuffle (Efraimidis-Spirakis) so heavier paths tend to come first.
            by_start[start].sort(key=lambda item: -(rng.random() ** (1.0 / max(item[1], 1e-6))))
        starts = sorted(by_start)
        rng.shuffle(starts)
        tasks: list[Task] = []
        seen: set[tuple[str, ...]] = set()
        progressed = True
        while len(tasks) < target and progressed:
            progressed = False
            for start in starts:
                queue = by_start[start]
                while queue:
                    path, _ = queue.pop(0)
                    if path in seen:
                        continue
                    seen.add(path)
                    progressed = True
                    task = self._try(path, 1)
                    if task is not None:
                        tasks.append(task)
                        break
                if len(tasks) >= target:
                    break
        return tasks

    def level2(self) -> list[Task]:
        rng = random.Random(stable_hash("level2", self.index.name, self.cfg.seed))
        flow = self.graphs.flow
        nodes = sorted(flow.nodes)
        sources = [
            fid for fid in nodes
            if self.index.functions[fid].receiver is None
            and not any(self.synth.is_object_type(p.type) for p in self.index.functions[fid].required_params)
        ] or nodes
        tasks: list[Task] = []
        seen: set[tuple[str, ...]] = set()
        min_calls = self.cfg.level2_min_calls
        attempts = 0
        budget = max(self.cfg.level2_tasks * 400, 2_000)
        while len(tasks) < self.cfg.level2_tasks and attempts < budget:
            attempts += 1
            if attempts == budget // 2 and not tasks and min_calls > 3:
                min_calls = 3  # sparse graphs may not support long compositions
            length = rng.randint(min_calls, max(self.cfg.level2_max_calls, min_calls))
            path = [rng.choice(sources)]
            while len(path) < length:
                options = [(dst, w) for dst, w in self.graphs.successors(path[-1]) if dst not in path]
                if not options:
                    break
                total = sum(w for _, w in options)
                pick = rng.random() * total
                for dst, w in options:
                    pick -= w
                    if pick <= 0:
                        path.append(dst)
                        break
                else:
                    path.append(options[-1][0])
            key = tuple(path)
            if len(path) < min_calls or key in seen:
                continue
            seen.add(key)
            task = self._try(key, 2)
            if task is not None:
                tasks.append(task)
        return tasks

    def generate(self) -> TaskSet:
        tasks = self.level0() + self.level1() + self.level2()
        taskset = TaskSet(tasks)
        log.info("tasks for %s: %s (%d paths skipped)", self.index.name, taskset.counts(), len(self.skipped))
        return taskset
