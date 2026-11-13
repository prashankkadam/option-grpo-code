"""Static analysis of SDK source code.

The SDK is never imported: every module is parsed with :mod:`ast`, which keeps the
analysis safe, fast and independent of optional runtime dependencies of the SDK.  The
result is an :class:`SDKIndex` holding every public function, method and constructor
with resolved parameter and return types, docstrings and source-level assertions.
"""

from __future__ import annotations

import ast
import builtins
import importlib.metadata
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from option_grpo.sdk.types import ANY, TypeRef, parse_annotation
from option_grpo.utils import get_logger, read_json, write_json

log = get_logger(__name__)

_DATACLASS_DECORATORS = {"dataclass", "define", "frozen", "mutable", "attrs", "s", "pydantic_dataclass"}
_MODEL_BASES = {"BaseModel", "BaseSettings", "NamedTuple", "TypedDict", "Struct", "SQLModel", "LanceModel"}
_ENUM_BASES = {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"}
_EXCEPTION_BASES = {"Exception", "BaseException", "Warning", "ValueError", "TypeError", "RuntimeError", "KeyError", "OSError", "IOError", "LookupError", "ArithmeticError", "AttributeError", "NotImplementedError"}
_PROTOCOL_BASES = {"Protocol"}
_SKIP_DECORATORS = {"overload"}
_BUILTIN_NAMES = frozenset(dir(builtins))


# --------------------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------------------
@dataclass
class ParamInfo:
    name: str
    type: TypeRef | None
    kind: str = "positional"  # positional | keyword | vararg | kwarg
    has_default: bool = False
    default: str | None = None  # source text of the default value

    @property
    def required(self) -> bool:
        return not self.has_default and self.kind in {"positional", "keyword"}

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.to_json() if self.type else None,
            "kind": self.kind,
            "has_default": self.has_default,
            "default": self.default,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ParamInfo:
        return cls(d["name"], TypeRef.from_json(d["type"]), d["kind"], d["has_default"], d.get("default"))


@dataclass
class FunctionInfo:
    """A callable of the SDK: free function, method, class/static method or constructor."""

    id: str  # fully qualified definition name, e.g. ``pkg.frame.Frame.where``
    name: str  # name used at the call site (the class name for constructors)
    module: str
    kind: str  # function | method | classmethod | staticmethod | constructor | property
    params: list[ParamInfo]
    returns: TypeRef | None
    owner: str | None = None  # qualified name of the owning class
    receiver: TypeRef | None = None  # type of ``self`` for instance methods / properties
    doc: str | None = None
    asserts: list[str] = field(default_factory=list)
    public_path: str | None = None  # shortest public dotted path, e.g. ``pkg.Frame.where``
    is_async: bool = False
    is_abstract: bool = False
    source_file: str | None = None
    lineno: int = 0
    end_lineno: int = 0

    @property
    def is_public(self) -> bool:
        return self.public_path is not None

    @property
    def required_params(self) -> list[ParamInfo]:
        return [p for p in self.params if p.required]

    def input_types(self) -> list[tuple[str, TypeRef | None]]:
        """``(name, type)`` of every value the call consumes, the receiver first."""
        out: list[tuple[str, TypeRef | None]] = []
        if self.receiver is not None:
            out.append(("self", self.receiver))
        out.extend((p.name, p.type) for p in self.params)
        return out

    def signature(self) -> str:
        parts = []
        for p in self.params:
            prefix = {"vararg": "*", "kwarg": "**"}.get(p.kind, "")
            text = f"{prefix}{p.name}"
            if p.type is not None:
                text += f": {p.type}"
            if p.has_default:
                text += f" = {p.default if p.default is not None else '...'}"
            parts.append(text)
        ret = f" -> {self.returns}" if self.returns is not None else ""
        return f"{self.public_path or self.id}({', '.join(parts)}){ret}"

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "module": self.module,
            "kind": self.kind,
            "params": [p.to_json() for p in self.params],
            "returns": self.returns.to_json() if self.returns else None,
            "owner": self.owner,
            "receiver": self.receiver.to_json() if self.receiver else None,
            "doc": self.doc,
            "asserts": self.asserts,
            "public_path": self.public_path,
            "is_async": self.is_async,
            "is_abstract": self.is_abstract,
            "source_file": self.source_file,
            "lineno": self.lineno,
            "end_lineno": self.end_lineno,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> FunctionInfo:
        return cls(
            id=d["id"],
            name=d["name"],
            module=d["module"],
            kind=d["kind"],
            params=[ParamInfo.from_json(p) for p in d["params"]],
            returns=TypeRef.from_json(d["returns"]),
            owner=d.get("owner"),
            receiver=TypeRef.from_json(d.get("receiver")),
            doc=d.get("doc"),
            asserts=list(d.get("asserts", [])),
            public_path=d.get("public_path"),
            is_async=d.get("is_async", False),
            is_abstract=d.get("is_abstract", False),
            source_file=d.get("source_file"),
            lineno=d.get("lineno", 0),
            end_lineno=d.get("end_lineno", 0),
        )


@dataclass
class ClassInfo:
    qualname: str
    name: str
    module: str
    bases: list[str]  # resolved qualified names (SDK or external) or raw names
    doc: str | None = None
    attributes: list[str] = field(default_factory=list)  # public fields and properties
    methods: list[str] = field(default_factory=list)  # ids of methods defined on the class
    public_path: str | None = None
    category: str = "class"  # class | exception | enum | protocol | model
    is_abstract: bool = False
    source_file: str | None = None
    lineno: int = 0
    end_lineno: int = 0

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ClassInfo:
        return cls(**d)


@dataclass
class SDKIndex:
    """Everything the later stages need to know about an SDK."""

    name: str
    import_name: str
    version: str | None
    root: str
    modules: list[str]
    functions: dict[str, FunctionInfo]
    classes: dict[str, ClassInfo]

    def __post_init__(self) -> None:
        self._mro_cache: dict[str, tuple[str, ...]] = {}
        self._by_name: dict[str, list[FunctionInfo]] | None = None

    # ------------------------------------------------------------------ hierarchy
    def mro(self, qualname: str) -> tuple[str, ...]:
        """Linearised ancestors of a class (depth-first, the class itself first)."""
        cached = self._mro_cache.get(qualname)
        if cached is not None:
            return cached
        order: list[str] = []
        stack = [qualname]
        seen: set[str] = set()
        while stack:
            current = stack.pop(0)
            if current in seen:
                continue
            seen.add(current)
            order.append(current)
            info = self.classes.get(current)
            if info is not None:
                stack = list(info.bases) + stack
        result = tuple(order)
        self._mro_cache[qualname] = result
        return result

    def is_subclass(self, child: str, parent: str) -> bool:
        return child != parent and parent in self.mro(child)

    # ------------------------------------------------------------------ lookups
    def public_functions(self, include_properties: bool = False) -> list[FunctionInfo]:
        """Callables that form the node set ``F`` of the graphs, in a stable order."""
        kinds = {"function", "method", "classmethod", "staticmethod", "constructor"}
        if include_properties:
            kinds.add("property")
        out = [f for f in self.functions.values() if f.is_public and f.kind in kinds and not f.is_abstract]
        out.sort(key=lambda f: f.id)
        return out

    def by_name(self, name: str) -> list[FunctionInfo]:
        if self._by_name is None:
            table: dict[str, list[FunctionInfo]] = {}
            for fn in self.public_functions(include_properties=True):
                table.setdefault(fn.name, []).append(fn)
            self._by_name = table
        return self._by_name.get(name, [])

    def find_method(self, class_qualname: str, name: str) -> FunctionInfo | None:
        """Resolve ``name`` on a class following its ancestors."""
        for ancestor in self.mro(class_qualname):
            fn = self.functions.get(f"{ancestor}.{name}")
            if fn is not None and fn.is_public:
                return fn
        return None

    def constructor(self, class_qualname: str) -> FunctionInfo | None:
        return self.functions.get(f"{class_qualname}.__init__")

    def callable_names(self) -> set[str]:
        """Call-site names of all public callables."""
        return {f.name for f in self.public_functions()}

    def bare_callable_names(self) -> set[str]:
        """Names callable without a receiver: free functions and classes."""
        return {f.name for f in self.public_functions() if f.kind in {"function", "constructor"}}

    def namespace_names(self) -> set[str]:
        """Names that may follow a dot without being a call: modules, classes, attributes."""
        names: set[str] = set()
        for module in self.modules:
            names.update(part for part in module.split(".") if not part.startswith("_"))
        for cls in self.classes.values():
            if cls.public_path is not None:
                names.add(cls.name)
                names.update(a for a in cls.attributes if not a.startswith("_"))
        return names

    def read_source(self, item: FunctionInfo | ClassInfo) -> str:
        """Source text of a function or class (empty when the file is unavailable)."""
        if not item.source_file or not item.lineno:
            return ""
        path = Path(self.root).parent / item.source_file
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[item.lineno - 1 : item.end_lineno])

    # ------------------------------------------------------------------ persistence
    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "import_name": self.import_name,
            "version": self.version,
            "root": self.root,
            "modules": self.modules,
            "functions": [f.to_json() for f in self.functions.values()],
            "classes": [c.to_json() for c in self.classes.values()],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> SDKIndex:
        functions = {f["id"]: FunctionInfo.from_json(f) for f in d["functions"]}
        classes = {c["qualname"]: ClassInfo.from_json(c) for c in d["classes"]}
        return cls(d["name"], d["import_name"], d.get("version"), d["root"], list(d["modules"]), functions, classes)

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> SDKIndex:
        return cls.from_json(read_json(path))


# --------------------------------------------------------------------------------------
# Per-module records built in the first pass
# --------------------------------------------------------------------------------------
@dataclass
class _ModuleRecord:
    name: str
    path: Path
    is_package: bool
    tree: ast.Module
    doc_tree: ast.Module | None = None  # the .py next to a .pyi stub
    classes: dict[str, ast.ClassDef] = field(default_factory=dict)
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = field(default_factory=dict)
    imports: dict[str, str] = field(default_factory=dict)  # local name -> absolute dotted target
    aliases: dict[str, str] = field(default_factory=dict)  # X = some.dotted.name
    type_aliases: dict[str, ast.expr] = field(default_factory=dict)
    typevars: dict[str, ast.expr | None] = field(default_factory=dict)  # name -> bound
    all: list[str] | None = None

    @property
    def package(self) -> str:
        return self.name if self.is_package else self.name.rpartition(".")[0]


_Symbol = tuple[str, Any]  # (kind, value)


def _decorator_names(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute):
            names.add(target.attr)
        elif isinstance(target, ast.Name):
            names.add(target.id)
    return names


def _dotted_name(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _looks_like_type_expr(node: ast.expr) -> bool:
    """Heuristic for module-level ``Alias = Union[A, B]`` / ``Alias = A | B`` statements."""
    if isinstance(node, ast.Subscript):
        head = _dotted_name(node.value)
        return head is not None and head.rsplit(".", 1)[-1] in {
            "Union", "Optional", "List", "Dict", "Tuple", "Sequence", "Iterable", "Mapping", "Callable",
            "Literal", "Annotated", "list", "dict", "tuple", "set", "Set", "Type", "type",
        }
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return all(
            isinstance(side, (ast.Name, ast.Attribute, ast.Subscript, ast.BinOp, ast.Constant))
            for side in (node.left, node.right)
        )
    return False


class SDKAnalyzer:
    """Builds an :class:`SDKIndex` from a package directory."""

    def __init__(
        self,
        import_name: str,
        root: str | Path | None = None,
        name: str | None = None,
        dist_name: str | None = None,
        exclude_dirs: list[str] | None = None,
        include_private: bool = False,
    ) -> None:
        self.import_name = import_name
        self.name = name or import_name
        self.dist_name = dist_name or import_name
        self.include_private = include_private
        self.exclude_dirs = set(exclude_dirs or [])
        self.root = Path(root).resolve() if root is not None else self._locate(import_name)
        self.modules: dict[str, _ModuleRecord] = {}
        self.classes: dict[str, ClassInfo] = {}
        self.functions: dict[str, FunctionInfo] = {}
        self._class_nodes: dict[str, tuple[_ModuleRecord, ast.ClassDef]] = {}
        self._alias_guard: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------ discovery
    @staticmethod
    def _locate(import_name: str) -> Path:
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            raise ModuleNotFoundError(
                f"cannot locate package {import_name!r}; install it or pass an explicit source root"
            )
        if spec.submodule_search_locations:
            return Path(next(iter(spec.submodule_search_locations))).resolve()
        if spec.origin:
            return Path(spec.origin).resolve()
        raise ModuleNotFoundError(f"package {import_name!r} has no source location")

    def _iter_sources(self) -> list[tuple[str, Path, bool]]:
        """``(module name, primary file, is_package)`` for every module under the root."""
        if self.root.is_file():
            return [(self.import_name, self.root, False)]
        found: dict[str, tuple[Path, bool]] = {}
        for path in sorted(self.root.rglob("*.py*")):
            if path.suffix not in {".py", ".pyi"}:
                continue
            rel = path.relative_to(self.root)
            if any(part in self.exclude_dirs or part.startswith(".") for part in rel.parts[:-1]):
                continue
            parts = [self.import_name, *rel.parts[:-1]]
            stem = path.name.split(".")[0]
            is_package = stem == "__init__"
            if not is_package:
                parts.append(stem)
            if not all(part.isidentifier() for part in parts):
                continue
            module = ".".join(parts)
            previous = found.get(module)
            # Prefer stubs for signatures; the matching .py is used for docs and asserts.
            if previous is None or path.suffix == ".pyi":
                found[module] = (path, is_package)
        return [(name, path, is_pkg) for name, (path, is_pkg) in sorted(found.items())]

    # ------------------------------------------------------------------ pass 1
    def _load_modules(self) -> None:
        for module, path, is_package in self._iter_sources():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
            except SyntaxError as exc:
                log.warning("skipping %s: %s", path, exc)
                continue
            record = _ModuleRecord(module, path, is_package, tree)
            if path.suffix == ".pyi":
                source = path.with_suffix(".py")
                if source.exists():
                    try:
                        record.doc_tree = ast.parse(source.read_text(encoding="utf-8", errors="replace"))
                    except SyntaxError:
                        record.doc_tree = None
            self._scan_body(record, tree.body)
            self.modules[module] = record

    def _absolute_import(self, record: _ModuleRecord, node: ast.ImportFrom) -> str:
        if node.level == 0:
            return node.module or ""
        base = record.package.split(".") if record.package else []
        if node.level > 1:
            base = base[: len(base) - (node.level - 1)]
        if node.module:
            base = base + node.module.split(".")
        return ".".join(base)

    def _scan_body(self, record: _ModuleRecord, body: list[ast.stmt]) -> None:
        for stmt in body:
            if isinstance(stmt, ast.ClassDef):
                record.classes[stmt.name] = stmt
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if "overload" in _decorator_names(stmt) and stmt.name in record.functions:
                    continue
                record.functions[stmt.name] = stmt
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if alias.asname:
                        record.imports[alias.asname] = alias.name
                    else:
                        head = alias.name.split(".")[0]
                        record.imports[head] = head
            elif isinstance(stmt, ast.ImportFrom):
                base = self._absolute_import(record, stmt)
                for alias in stmt.names:
                    if alias.name == "*":
                        record.imports.setdefault(f"*{base}", base)
                        continue
                    record.imports[alias.asname or alias.name] = f"{base}.{alias.name}" if base else alias.name
            elif isinstance(stmt, (ast.If, ast.Try)):
                # ``if TYPE_CHECKING:`` and guarded optional imports contribute symbols too.
                self._scan_body(record, stmt.body)
                for handler in getattr(stmt, "handlers", []):
                    self._scan_body(record, handler.body)
                self._scan_body(record, stmt.orelse)
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                self._scan_assign(record, stmt.targets[0].id, stmt.value)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                annotation = _dotted_name(stmt.annotation) or ""
                if annotation.rsplit(".", 1)[-1] == "TypeAlias":
                    record.type_aliases[stmt.target.id] = stmt.value
                else:
                    self._scan_assign(record, stmt.target.id, stmt.value)
            elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "__all__":
                extra = self._string_list(stmt.value)
                if extra is not None and record.all is not None:
                    record.all.extend(extra)
            elif hasattr(ast, "TypeAlias") and isinstance(stmt, ast.TypeAlias):  # pragma: no cover - 3.12+
                if isinstance(stmt.name, ast.Name):
                    record.type_aliases[stmt.name.id] = stmt.value

    @staticmethod
    def _string_list(node: ast.expr) -> list[str] | None:
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if len(values) == len(node.elts):
                return values
        return None

    def _scan_assign(self, record: _ModuleRecord, name: str, value: ast.expr) -> None:
        if name == "__all__":
            record.all = self._string_list(value)
            return
        if isinstance(value, ast.Call):
            callee = _dotted_name(value.func) or ""
            short = callee.rsplit(".", 1)[-1]
            if short in {"TypeVar", "ParamSpec", "TypeVarTuple"}:
                bound = next((kw.value for kw in value.keywords if kw.arg == "bound"), None)
                record.typevars[name] = bound
            elif short == "NewType" and len(value.args) == 2:
                record.type_aliases[name] = value.args[1]
            return
        dotted = _dotted_name(value)
        if dotted is not None:
            record.aliases[name] = dotted
        elif _looks_like_type_expr(value):
            record.type_aliases[name] = value

    # ------------------------------------------------------------------ resolution
    def _resolve_in_module(self, module: str, name: str, seen: set[tuple[str, str]]) -> _Symbol | None:
        record = self.modules.get(module)
        if record is None or (module, name) in seen:
            return None
        seen.add((module, name))
        if name in record.classes:
            return ("class", f"{module}.{name}")
        if name in record.functions:
            return ("function", f"{module}.{name}")
        if name in record.typevars:
            return ("typevar", (module, record.typevars[name]))
        if name in record.type_aliases:
            return ("typealias", (module, name))
        if name in record.imports:
            return self._resolve_absolute(record.imports[name], seen)
        if name in record.aliases:
            return self._resolve_dotted(module, record.aliases[name], seen)
        if f"{module}.{name}" in self.modules:
            return ("module", f"{module}.{name}")
        for key, target in record.imports.items():  # star imports
            if key.startswith("*") and target in self.modules:
                star = self.modules[target]
                if star.all is not None and name not in star.all:
                    continue
                found = self._resolve_in_module(target, name, seen)
                if found is not None:
                    return found
        return None

    def _attr(self, symbol: _Symbol, attr: str, seen: set[tuple[str, str]]) -> _Symbol | None:
        kind, value = symbol
        if kind == "module":
            return self._resolve_in_module(value, attr, seen)
        if kind == "class":
            nested = f"{value}.{attr}"
            if nested in self._class_nodes:
                return ("class", nested)
            return None
        if kind == "external":
            return ("external", f"{value}.{attr}")
        return None

    def _resolve_absolute(self, dotted: str, seen: set[tuple[str, str]]) -> _Symbol | None:
        parts = dotted.split(".")
        for cut in range(len(parts), 0, -1):
            module = ".".join(parts[:cut])
            if module in self.modules:
                symbol: _Symbol | None = ("module", module)
                for attr in parts[cut:]:
                    symbol = self._attr(symbol, attr, seen)  # type: ignore[arg-type]
                    if symbol is None:
                        return None
                return symbol
        if parts[0] == self.import_name:
            return None  # refers to an SDK module we could not parse
        return ("external", dotted)

    def _resolve_dotted(self, module: str, dotted: str, seen: set[tuple[str, str]] | None = None) -> _Symbol | None:
        seen = set() if seen is None else seen
        head, *rest = dotted.split(".")
        symbol = self._resolve_in_module(module, head, seen)
        if symbol is None:
            return None
        for attr in rest:
            symbol = self._attr(symbol, attr, seen)
            if symbol is None:
                return None
        return symbol

    def _resolver(self, module: str):
        def resolve(dotted: str) -> tuple[str, bool] | TypeRef | None:
            symbol = self._resolve_dotted(module, dotted)
            if symbol is None:
                return None
            kind, value = symbol
            if kind == "class":
                return (value, True)
            if kind == "external":
                return (value, False)
            if kind == "typevar":
                tv_module, bound = value
                if bound is None:
                    return ANY
                return parse_annotation(bound, self._resolver(tv_module)) or ANY
            if kind == "typealias":
                alias_module, alias_name = value
                if (alias_module, alias_name) in self._alias_guard:
                    return ANY
                self._alias_guard.add((alias_module, alias_name))
                try:
                    expr = self.modules[alias_module].type_aliases[alias_name]
                    return parse_annotation(expr, self._resolver(alias_module)) or ANY
                finally:
                    self._alias_guard.discard((alias_module, alias_name))
            return None

        return resolve

    # ------------------------------------------------------------------ pass 2
    def _register_classes(self) -> None:
        def visit(record: _ModuleRecord, node: ast.ClassDef, prefix: str) -> None:
            qualname = f"{prefix}.{node.name}"
            self._class_nodes[qualname] = (record, node)
            for child in node.body:
                if isinstance(child, ast.ClassDef):
                    visit(record, child, qualname)

        for record in self.modules.values():
            for node in record.classes.values():
                visit(record, node, record.name)

    def _doc_lookup(self, record: _ModuleRecord) -> dict[str, ast.AST]:
        """Map ``Class.method`` / ``function`` names to nodes of the .py behind a stub."""
        table: dict[str, ast.AST] = {}
        if record.doc_tree is None:
            return table
        for stmt in record.doc_tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                table[stmt.name] = stmt
            elif isinstance(stmt, ast.ClassDef):
                table[stmt.name] = stmt
                for child in stmt.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        table[f"{stmt.name}.{child.name}"] = child
        return table

    def _build_class(self, qualname: str) -> None:
        record, node = self._class_nodes[qualname]
        resolver = self._resolver(record.name)
        bases: list[str] = []
        raw_bases: list[str] = []
        for base in node.bases:
            target = base.value if isinstance(base, ast.Subscript) else base  # Generic[T], Base[T]
            dotted = _dotted_name(target)
            if dotted is None:
                continue
            raw_bases.append(dotted.rsplit(".", 1)[-1])
            resolved = resolver(dotted)
            if isinstance(resolved, tuple):
                bases.append(resolved[0])
            else:
                bases.append(dotted)
        decorators = _decorator_names(node)
        category = "class"
        if set(raw_bases) & _ENUM_BASES:
            category = "enum"
        elif set(raw_bases) & _PROTOCOL_BASES:
            category = "protocol"
        elif set(raw_bases) & _EXCEPTION_BASES or node.name.endswith(("Error", "Exception", "Warning")):
            category = "exception"
        elif decorators & _DATACLASS_DECORATORS or set(raw_bases) & _MODEL_BASES:
            category = "model"
        attributes: list[str] = []
        is_abstract = False
        for child in node.body:
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                attributes.append(child.target.id)
            elif isinstance(child, ast.Assign):
                attributes.extend(t.id for t in child.targets if isinstance(t, ast.Name))
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = _decorator_names(child)
                if names & {"property", "cached_property", "functools.cached_property"}:
                    attributes.append(child.name)
                if "abstractmethod" in names:
                    is_abstract = True
                if child.name == "__init__":
                    for sub in ast.walk(child):
                        if isinstance(sub, (ast.Assign, ast.AnnAssign)):
                            targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                            for t in targets:
                                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                                    attributes.append(t.attr)
        rel = self._relative(record.path)
        self.classes[qualname] = ClassInfo(
            qualname=qualname,
            name=node.name,
            module=record.name,
            bases=bases,
            doc=ast.get_docstring(node),
            attributes=sorted({a for a in attributes if not a.startswith("_")}),
            category=category,
            is_abstract=is_abstract,
            source_file=rel,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno) or node.lineno,
        )

    def _relative(self, path: Path) -> str:
        base = self.root.parent
        try:
            return str(path.relative_to(base))
        except ValueError:
            return str(path)

    def _params(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        module: str,
        self_type: TypeRef | None,
        drop_first: bool,
    ) -> list[ParamInfo]:
        resolver = self._resolver(module)
        args = node.args
        out: list[ParamInfo] = []
        positional = list(args.posonlyargs) + list(args.args)
        defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
        for index, (arg, default) in enumerate(zip(positional, defaults)):
            if drop_first and index == 0:
                continue
            out.append(self._param(arg, default, "positional", resolver, self_type))
        if args.vararg is not None:
            out.append(self._param(args.vararg, None, "vararg", resolver, self_type))
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            out.append(self._param(arg, default, "keyword", resolver, self_type))
        if args.kwarg is not None:
            out.append(self._param(args.kwarg, None, "kwarg", resolver, self_type))
        return out

    @staticmethod
    def _param(arg: ast.arg, default: ast.expr | None, kind: str, resolver, self_type: TypeRef | None) -> ParamInfo:
        default_src = None
        if default is not None:
            try:
                default_src = ast.unparse(default)
            except Exception:  # pragma: no cover - defensive
                default_src = "..."
        return ParamInfo(
            name=arg.arg,
            type=parse_annotation(arg.annotation, resolver, self_type),
            kind=kind,
            has_default=default is not None,
            default=default_src,
        )

    @staticmethod
    def _asserts(node: ast.FunctionDef | ast.AsyncFunctionDef, param_names: set[str]) -> list[str]:
        """``assert`` expressions over the parameters only (Algorithm 3, line 9)."""
        out: list[str] = []

        def walk(stmts: list[ast.stmt]) -> None:
            for stmt in stmts:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(stmt, ast.Assert):
                    names = {n.id for n in ast.walk(stmt.test) if isinstance(n, ast.Name)}
                    free = names - param_names - _BUILTIN_NAMES
                    if names & param_names and not free:
                        out.append(ast.unparse(stmt.test))
                    continue
                if isinstance(stmt, (ast.Return, ast.Raise)):
                    return  # only preconditions that precede the first exit
                if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                    for t in targets:  # a rebound parameter no longer reflects the argument
                        for n in ast.walk(t):
                            if isinstance(n, ast.Name):
                                param_names.discard(n.id)
        walk(node.body)
        return out

    def _build_function(
        self,
        record: _ModuleRecord,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        owner: str | None,
        doc_nodes: dict[str, ast.AST],
    ) -> None:
        decorators = _decorator_names(node)
        qualprefix = owner or record.name
        fid = f"{qualprefix}.{node.name}"
        if "overload" in decorators and fid in self.functions:
            return
        if decorators & {"setter", "deleter", "getter"}:
            return
        self_type: TypeRef | None = None
        receiver: TypeRef | None = None
        kind = "function"
        drop_first = False
        call_name = node.name
        if owner is not None:
            cls = self.classes[owner]
            self_type = TypeRef(cls.name, qualname=owner, sdk=True)
            if "staticmethod" in decorators:
                kind = "staticmethod"
            elif "classmethod" in decorators:
                kind, drop_first = "classmethod", True
            elif decorators & {"property", "cached_property"}:
                kind, drop_first, receiver = "property", True, self_type
            elif node.name == "__init__":
                kind, drop_first, call_name = "constructor", True, cls.name
            elif node.name == "__new__":
                return
            else:
                kind, drop_first, receiver = "method", True, self_type
        params = self._params(node, record.name, self_type, drop_first)
        returns = parse_annotation(node.returns, self._resolver(record.name), self_type)
        if kind == "constructor":
            returns = self_type
        doc_key = f"{self.classes[owner].name}.{node.name}" if owner is not None else node.name
        doc_node = doc_nodes.get(doc_key)
        doc = ast.get_docstring(node)
        asserts_node = node
        if doc_node is not None and isinstance(doc_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = doc or ast.get_docstring(doc_node)
            asserts_node = doc_node
        if kind == "constructor" and not doc and owner is not None:
            doc = self.classes[owner].doc
        self.functions[fid] = FunctionInfo(
            id=fid,
            name=call_name,
            module=record.name,
            kind=kind,
            params=params,
            returns=returns,
            owner=owner,
            receiver=receiver,
            doc=doc,
            asserts=self._asserts(asserts_node, {p.name for p in params}),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_abstract="abstractmethod" in decorators,
            source_file=self._relative(record.path),
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno) or node.lineno,
        )

    def _synthesize_constructor(self, qualname: str) -> None:
        """Constructors of dataclass-like models come from their annotated fields."""
        cls = self.classes[qualname]
        record, node = self._class_nodes[qualname]
        resolver = self._resolver(record.name)
        self_type = TypeRef(cls.name, qualname=qualname, sdk=True)
        params: list[ParamInfo] = []
        for ancestor in reversed(self.mro(qualname)):
            if ancestor not in self._class_nodes or self.classes[ancestor].category != "model":
                continue
            anc_record, anc_node = self._class_nodes[ancestor]
            anc_resolver = resolver if ancestor == qualname else self._resolver(anc_record.name)
            for child in anc_node.body:
                if not (isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)):
                    continue
                name = child.target.id
                annotation = _dotted_name(child.annotation.value if isinstance(child.annotation, ast.Subscript) else child.annotation)
                if name.startswith("_") or (annotation or "").rsplit(".", 1)[-1] == "ClassVar":
                    continue
                params = [p for p in params if p.name != name]
                params.append(
                    ParamInfo(
                        name=name,
                        type=parse_annotation(child.annotation, anc_resolver, self_type),
                        kind="keyword",
                        has_default=child.value is not None,
                        default=ast.unparse(child.value) if child.value is not None else None,
                    )
                )
        # Fields with defaults may not precede required ones in the call; order them.
        params.sort(key=lambda p: p.has_default)
        self.functions[f"{qualname}.__init__"] = FunctionInfo(
            id=f"{qualname}.__init__",
            name=cls.name,
            module=cls.module,
            kind="constructor",
            params=params,
            returns=self_type,
            owner=qualname,
            doc=cls.doc,
            source_file=cls.source_file,
            lineno=node.lineno,
            end_lineno=cls.end_lineno,
        )

    def mro(self, qualname: str) -> tuple[str, ...]:
        order: list[str] = []
        stack = [qualname]
        seen: set[str] = set()
        while stack:
            current = stack.pop(0)
            if current in seen:
                continue
            seen.add(current)
            order.append(current)
            info = self.classes.get(current)
            if info is not None:
                stack = list(info.bases) + stack
        return tuple(order)

    def _build_functions(self) -> None:
        for record in self.modules.values():
            doc_nodes = self._doc_lookup(record)
            for node in record.functions.values():
                self._build_function(record, node, None, doc_nodes)
        for qualname, (record, node) in self._class_nodes.items():
            doc_nodes = self._doc_lookup(record)
            cls = self.classes[qualname]
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._build_function(record, child, qualname, doc_nodes)
                    fid = f"{qualname}.{child.name}"
                    if fid in self.functions and fid not in cls.methods:
                        cls.methods.append(fid)
        # Constructors that are not written out explicitly.
        for qualname, cls in self.classes.items():
            if f"{qualname}.__init__" in self.functions:
                continue
            if cls.category == "model":
                self._synthesize_constructor(qualname)
                continue
            if cls.category != "class":
                continue
            for ancestor in self.mro(qualname)[1:]:
                inherited = self.functions.get(f"{ancestor}.__init__")
                if inherited is not None:
                    self_type = TypeRef(cls.name, qualname=qualname, sdk=True)
                    self.functions[f"{qualname}.__init__"] = FunctionInfo(
                        id=f"{qualname}.__init__",
                        name=cls.name,
                        module=cls.module,
                        kind="constructor",
                        params=list(inherited.params),
                        returns=self_type,
                        owner=qualname,
                        doc=cls.doc or inherited.doc,
                        asserts=list(inherited.asserts),
                        source_file=cls.source_file,
                        lineno=cls.lineno,
                        end_lineno=cls.end_lineno,
                    )
                    break
        for qualname, cls in self.classes.items():
            ctor = self.functions.get(f"{qualname}.__init__")
            if ctor is None:
                continue
            if cls.category in {"exception", "enum", "protocol"} or cls.is_abstract:
                ctor.is_abstract = True  # never proposed as a call target

    # ------------------------------------------------------------------ pass 3
    @staticmethod
    def _is_public_module(module: str) -> bool:
        return not any(part.startswith("_") for part in module.split("."))

    def _assign_public_paths(self) -> None:
        candidates: dict[str, list[str]] = {}

        def offer(qualified: str, path: str) -> None:
            candidates.setdefault(qualified, []).append(path)

        for record in self.modules.values():
            if not (self.include_private or self._is_public_module(record.name)):
                continue
            names: set[str] = set(record.classes) | set(record.functions)
            if record.all is not None:
                names = set(record.all)
            else:
                if record.is_package:
                    names |= {n for n in record.imports if not n.startswith("*")}
                    names |= set(record.aliases)
                names = {n for n in names if self.include_private or not n.startswith("_")}
            for name in names:
                symbol = self._resolve_in_module(record.name, name, set())
                if symbol is not None and symbol[0] in {"class", "function"}:
                    offer(symbol[1], f"{record.name}.{name}")

        def best(paths: list[str]) -> str:
            return min(paths, key=lambda p: (p.count("."), len(p), p))

        for qualname, cls in self.classes.items():
            if qualname in candidates:
                cls.public_path = best(candidates[qualname])
        # Nested classes are public through their (public) parent class.
        for qualname in sorted(self.classes, key=lambda q: q.count(".")):
            cls = self.classes[qualname]
            parent = qualname.rpartition(".")[0]
            if cls.public_path is None and parent in self.classes and self.classes[parent].public_path:
                if self.include_private or not cls.name.startswith("_"):
                    cls.public_path = f"{self.classes[parent].public_path}.{cls.name}"
        for fid, fn in self.functions.items():
            if fn.owner is None:
                if fid in candidates:
                    fn.public_path = best(candidates[fid])
                continue
            owner = self.classes.get(fn.owner)
            if owner is None or owner.public_path is None:
                continue
            if fn.kind == "constructor":
                fn.public_path = owner.public_path
            elif self.include_private or not fn.name.startswith("_"):
                fn.public_path = f"{owner.public_path}.{fn.name}"

    # ------------------------------------------------------------------ entry point
    def analyze(self) -> SDKIndex:
        self._load_modules()
        if not self.modules:
            raise RuntimeError(f"no Python sources found under {self.root}")
        self._register_classes()
        for qualname in self._class_nodes:
            self._build_class(qualname)
        self._build_functions()
        self._assign_public_paths()
        try:
            version: str | None = importlib.metadata.version(self.dist_name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        index = SDKIndex(
            name=self.name,
            import_name=self.import_name,
            version=version,
            root=str(self.root),
            modules=sorted(m for m in self.modules if self.include_private or self._is_public_module(m)),
            functions=dict(sorted(self.functions.items())),
            classes=dict(sorted(self.classes.items())),
        )
        log.info(
            "indexed %s: %d modules, %d classes, %d public callables",
            self.name, len(self.modules), len(self.classes), len(index.public_functions()),
        )
        return index


def analyze_sdk(
    import_name: str,
    root: str | Path | None = None,
    name: str | None = None,
    dist_name: str | None = None,
    exclude_dirs: list[str] | None = None,
    include_private: bool = False,
) -> SDKIndex:
    """Convenience wrapper around :class:`SDKAnalyzer`."""
    return SDKAnalyzer(import_name, root, name, dist_name, exclude_dirs, include_private).analyze()

