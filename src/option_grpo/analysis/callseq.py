"""Static extraction of SDK API-call sequences from generated programs.

Programs are parsed with :mod:`ast` and calls are resolved against the SDK index with a
light forward type propagation (``df = sdk.read_csv(p)`` gives ``df`` the annotated
return type of ``read_csv``, so ``df.where`` resolves to the right method).  Calls are
reported in evaluation order and tagged as *chain* calls (not nested inside another
call's arguments) or *arg* calls (argument constructors).
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from typing import Any

from option_grpo.sdk.introspect import ClassInfo, FunctionInfo, SDKIndex
from option_grpo.sdk.types import TypeRef

_STDLIB = frozenset(getattr(sys, "stdlib_module_names", ())) | {"__future__"}


@dataclass
class CallSite:
    name: str  # name written at the call site
    func: str | None  # id of the resolved SDK function, when unambiguous
    sdk: bool  # the call targets the SDK
    role: str  # "chain" | "arg"
    depth: int  # number of enclosing call-argument lists
    stmt: int  # index of the enclosing top-level statement
    lineno: int
    col: int
    end_lineno: int
    end_col: int
    foreign: str | None = None  # third-party module the call targets, if any
    builtin: bool = False
    name_lineno: int = 0  # where the callee name starts (the dot of an attribute call)
    name_col: int = 0


@dataclass
class ProgramAnalysis:
    ok: bool
    error: str | None = None
    calls: list[CallSite] = field(default_factory=list)
    imports: dict[str, str] = field(default_factory=dict)  # alias -> module introduced by the solution
    foreign_modules: set[str] = field(default_factory=set)  # third-party modules actually called
    tree: ast.Module | None = None

    @property
    def sdk_calls(self) -> list[CallSite]:
        return [c for c in self.calls if c.sdk]

    @property
    def primary(self) -> list[CallSite]:
        """SDK calls on the main call chains, in evaluation order."""
        return [c for c in self.calls if c.sdk and c.role == "chain"]

    def primary_names(self) -> list[str]:
        return [c.name for c in self.primary]

    @property
    def n_api(self) -> int:
        """``N_api``: number of SDK API calls in the program."""
        return len(self.sdk_calls)


# A statically inferred value: ("type", TypeRef) | ("module", dotted) | ("class", qualname)
# | ("func", FunctionInfo) | ("foreign", module) | ("builtin", name) | None
_Value = tuple[str, Any] | None


class _Resolver:
    def __init__(self, index: SDKIndex) -> None:
        self.index = index
        self.by_path: dict[str, FunctionInfo | ClassInfo] = {}
        self.modules = set(index.modules)
        for cls in index.classes.values():
            self.by_path[cls.qualname] = cls
            if cls.public_path:
                self.by_path[cls.public_path] = cls
        for fn in index.functions.values():
            if not fn.is_public or fn.kind == "constructor":
                continue
            self.by_path.setdefault(fn.id, fn)
            if fn.public_path:
                self.by_path[fn.public_path] = fn
        self.method_names = {
            fn.name for fn in index.public_functions(include_properties=True) if fn.kind in {"method", "property"}
        }

    def module_attr(self, module: str, attr: str) -> _Value:
        dotted = f"{module}.{attr}"
        target = self.by_path.get(dotted)
        if isinstance(target, ClassInfo):
            return ("class", target.qualname)
        if isinstance(target, FunctionInfo):
            return ("func", target)
        if dotted in self.modules or any(m.startswith(dotted + ".") for m in self.modules):
            return ("module", dotted)
        return None

    def class_attr(self, qualname: str, attr: str) -> _Value:
        fn = self.index.find_method(qualname, attr)
        if fn is not None:
            if fn.kind == "property":
                return ("type", fn.returns) if fn.returns is not None else None
            return ("func", fn)
        nested = self.index.classes.get(f"{qualname}.{attr}")
        if nested is not None:
            return ("class", nested.qualname)
        return None

    def type_attr(self, ref: TypeRef, attr: str) -> _Value:
        for alt in ref.alternatives():
            if alt.sdk and alt.qualname:
                found = self.class_attr(alt.qualname, attr)
                if found is not None:
                    return found
        return None

    def is_sdk_type(self, ref: TypeRef) -> bool:
        return any(alt.sdk for alt in ref.alternatives())


class _Analyzer(ast.NodeVisitor):
    def __init__(self, index: SDKIndex, resolver: _Resolver) -> None:
        self.index = index
        self.resolver = resolver
        self.env: dict[str, _Value] = {}
        self.calls: list[CallSite] = []
        self.imports: dict[str, str] = {}
        self.foreign: set[str] = set()
        self.recording = True
        self.depth = 0
        self.stmt = 0

    # ------------------------------------------------------------------ imports
    def _bind_module(self, alias: str, module: str) -> None:
        root = module.split(".")[0]
        if root == self.index.import_name:
            self.env[alias] = ("module", module)
        elif root in _STDLIB:
            self.env[alias] = ("builtin", module)
        else:
            self.env[alias] = ("foreign", root)
        if self.recording:
            self.imports[alias] = module

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.asname:
                self._bind_module(alias.asname, alias.name)
            else:
                self._bind_module(alias.name.split(".")[0], alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".")[0]
        for alias in node.names:
            local = alias.asname or alias.name
            if root == self.index.import_name:
                value = self.resolver.module_attr(module, alias.name)
                self.env[local] = value
            elif root in _STDLIB:
                self.env[local] = ("builtin", f"{module}.{alias.name}")
            else:
                self.env[local] = ("foreign", root)
            if self.recording:
                self.imports[local] = f"{module}.{alias.name}"

    # ------------------------------------------------------------------ inference
    def infer(self, node: ast.expr | None) -> _Value:
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return self.env.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self.infer(node.value)
            if base is None:
                return None
            kind, value = base
            if kind == "module":
                return self.resolver.module_attr(value, node.attr)
            if kind == "class":
                return self.resolver.class_attr(value, node.attr)
            if kind == "type":
                return self.resolver.type_attr(value, node.attr)
            if kind in {"foreign", "builtin"}:
                return (kind, value)
            return None
        if isinstance(node, ast.Call):
            return self._call_value(self.infer(node.func))
        if isinstance(node, ast.Await):
            return self.infer(node.value)
        if isinstance(node, ast.Constant):
            return ("type", TypeRef(type(node.value).__name__ if node.value is not None else "None"))
        if isinstance(node, (ast.List, ast.ListComp)):
            return ("type", TypeRef("list"))
        if isinstance(node, (ast.Dict, ast.DictComp)):
            return ("type", TypeRef("dict"))
        if isinstance(node, (ast.Tuple,)):
            return ("type", TypeRef("tuple"))
        if isinstance(node, (ast.Set, ast.SetComp)):
            return ("type", TypeRef("set"))
        if isinstance(node, ast.JoinedStr):
            return ("type", TypeRef("str"))
        if isinstance(node, ast.IfExp):
            return self.infer(node.body) or self.infer(node.orelse)
        if isinstance(node, ast.NamedExpr):
            value = self.infer(node.value)
            self.env[node.target.id] = value
            return value
        if isinstance(node, ast.BinOp):
            left = self.infer(node.left)
            # Operator overloads of SDK expression types usually stay within the type.
            if left is not None and left[0] == "type" and self.resolver.is_sdk_type(left[1]):
                return left
            right = self.infer(node.right)
            if right is not None and right[0] == "type" and self.resolver.is_sdk_type(right[1]):
                return right
            return None
        if isinstance(node, ast.Compare):
            left = self.infer(node.left)
            if left is not None and left[0] == "type" and self.resolver.is_sdk_type(left[1]):
                return left
            return ("type", TypeRef("bool"))
        return None

    @staticmethod
    def _call_value(callee: _Value) -> _Value:
        if callee is None:
            return None
        kind, value = callee
        if kind == "func":
            return ("type", value.returns) if value.returns is not None else None
        if kind == "class":
            return ("type", TypeRef(value.rsplit(".", 1)[-1], qualname=value, sdk=True))
        if kind == "foreign":
            return ("foreign", value)
        return None

    # ------------------------------------------------------------------ calls
    def visit_Call(self, node: ast.Call) -> None:
        # Evaluation order: receiver chain, then arguments, then the call itself.
        self.visit(node.func)
        self.depth += 1
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)
        self.depth -= 1
        if not self.recording:
            return
        callee = self.infer(node.func)
        name = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else ""
        )
        func_id: str | None = None
        sdk = False
        foreign: str | None = None
        builtin = False
        if callee is not None:
            kind, value = callee
            if kind == "func":
                func_id, sdk, name = value.id, True, value.name
            elif kind == "class":
                ctor = self.index.constructor(value)
                func_id = ctor.id if ctor is not None else None
                sdk = True
                name = value.rsplit(".", 1)[-1]
            elif kind == "foreign":
                foreign = value
                self.foreign.add(value)
            elif kind == "builtin":
                builtin = True
        elif isinstance(node.func, ast.Attribute):
            receiver = self.infer(node.func.value)
            if receiver is None:
                # Unknown receiver: fall back to the method name.
                candidates = [f for f in self.index.by_name(name) if f.kind == "method"]
                if candidates:
                    sdk = True
                    func_id = candidates[0].id if len(candidates) == 1 else None
            elif receiver[0] == "foreign":
                foreign = receiver[1]
                self.foreign.add(receiver[1])
            elif receiver[0] == "type" and not self.resolver.is_sdk_type(receiver[1]):
                builtin = True
        elif isinstance(node.func, ast.Name):
            builtin = True
        if not name:
            return
        if isinstance(node.func, ast.Attribute):
            anchor = node.func.value
            name_lineno = getattr(anchor, "end_lineno", node.lineno) or node.lineno
            name_col = getattr(anchor, "end_col_offset", node.col_offset) or node.col_offset
        else:
            name_lineno, name_col = node.lineno, node.col_offset
        self.calls.append(
            CallSite(
                name=name,
                func=func_id,
                sdk=sdk,
                role="chain" if self.depth == 0 else "arg",
                depth=self.depth,
                stmt=self.stmt,
                lineno=node.lineno,
                col=node.col_offset,
                end_lineno=getattr(node, "end_lineno", node.lineno) or node.lineno,
                end_col=getattr(node, "end_col_offset", node.col_offset) or node.col_offset,
                foreign=foreign,
                builtin=builtin,
                name_lineno=name_lineno,
                name_col=name_col,
            )
        )

    # ------------------------------------------------------------------ bindings
    def _bind(self, target: ast.expr, value: _Value) -> None:
        if isinstance(target, ast.Name):
            self.env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._bind(element, None)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        value = self.infer(node.value)
        for target in node.targets:
            self._bind(target, value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            self._bind(node.target, self.infer(node.value))

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._bind(item.optional_vars, self.infer(item.context_expr))
        for stmt in node.body:
            self.visit(stmt)

    visit_AsyncWith = visit_With  # type: ignore[assignment]

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._bind(node.target, None)
        for stmt in node.body + node.orelse:
            self.visit(stmt)

    visit_AsyncFor = visit_For  # type: ignore[assignment]

    def run(self, tree: ast.Module, record: bool) -> None:
        self.recording = record
        for position, stmt in enumerate(tree.body):
            self.stmt = position
            self.depth = 0
            self.visit(stmt)


def analyze_program(code: str, index: SDKIndex, context: str = "") -> ProgramAnalysis:
    """Analyse ``code``; ``context`` (prompt imports and setup) only seeds names and types."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError) as exc:
        return ProgramAnalysis(ok=False, error=f"{type(exc).__name__}: {exc}")
    analyzer = _Analyzer(index, _resolver_for(index))
    if context:
        try:
            analyzer.run(ast.parse(context), record=False)
        except (SyntaxError, ValueError):
            pass
    try:
        analyzer.run(tree, record=True)
    except RecursionError as exc:
        return ProgramAnalysis(ok=False, error=f"RecursionError: {exc}")
    return ProgramAnalysis(
        ok=True,
        calls=analyzer.calls,
        imports=analyzer.imports,
        foreign_modules=analyzer.foreign,
        tree=tree,
    )


_RESOLVERS: dict[int, tuple[SDKIndex, _Resolver]] = {}


def _resolver_for(index: SDKIndex) -> _Resolver:
    cached = _RESOLVERS.get(id(index))
    if cached is None or cached[0] is not index:
        cached = (index, _Resolver(index))
        _RESOLVERS[id(index)] = cached
    return cached[1]


def api_call_sequences(analysis: ProgramAnalysis) -> list[str]:
    """The API-call sequence of a program as used by Algorithm 2 (chain calls only)."""
    return analysis.primary_names()


def contiguous_subsequences(sequence: list[str], min_len: int, max_len: int) -> set[tuple[str, ...]]:
    """All distinct contiguous sub-sequences of the given lengths."""
    out: set[tuple[str, ...]] = set()
    for length in range(min_len, max_len + 1):
        for start in range(0, len(sequence) - length + 1):
            out.add(tuple(sequence[start : start + length]))
    return out


def char_offset(code: str, lineno: int, col: int) -> int:
    """Character offset of an AST position (``col`` is a UTF-8 byte offset within the line)."""
    lines = code.splitlines(keepends=True)
    if lineno < 1 or lineno > len(lines):
        return len(code)
    start = sum(len(line) for line in lines[: lineno - 1])
    prefix = lines[lineno - 1].encode("utf-8")[:col].decode("utf-8", errors="ignore")
    return start + len(prefix)
