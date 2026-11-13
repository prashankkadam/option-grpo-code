"""Type references parsed from annotations and the type-compatibility score.

``type_compat(f_a, f_b)`` in Algorithm 1 is the maximum, over the parameters of
``f_b`` (including the receiver of a method), of :func:`type_compat` between the return
type of ``f_a`` and the parameter type.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

# Canonical spellings of typing aliases.
_ALIASES = {
    "List": "list",
    "Dict": "dict",
    "Tuple": "tuple",
    "Set": "set",
    "FrozenSet": "frozenset",
    "Type": "type",
    "Text": "str",
    "AnyStr": "str",
    "LiteralString": "str",
    "NoneType": "None",
    "Sequence": "Sequence",
    "MutableSequence": "Sequence",
    "Collection": "Collection",
    "Iterable": "Iterable",
    "Iterator": "Iterator",
    "Generator": "Iterator",
    "AsyncIterator": "AsyncIterator",
    "AsyncIterable": "AsyncIterator",
    "AsyncGenerator": "AsyncIterator",
    "Mapping": "Mapping",
    "MutableMapping": "Mapping",
    "DefaultDict": "dict",
    "OrderedDict": "dict",
    "Deque": "Sequence",
    "Awaitable": "Awaitable",
    "Coroutine": "Awaitable",
    "PathLike": "Path",
    "PurePath": "Path",
    "StrPath": "Path",
}

PRIMITIVES = frozenset(
    {"str", "int", "float", "bool", "bytes", "bytearray", "complex", "None", "Path", "object", "date", "datetime", "timedelta", "Decimal", "UUID"}
)
CONTAINERS = frozenset(
    {"list", "dict", "tuple", "set", "frozenset", "Sequence", "Collection", "Iterable", "Iterator", "Mapping", "AsyncIterator"}
)
_SEQUENCE_LIKE = {"list", "tuple", "Sequence", "Collection", "Iterable", "Iterator", "set", "frozenset"}
_MAPPING_LIKE = {"dict", "Mapping"}
# Which container heads accept which (source -> acceptable destinations).
_CONTAINER_ACCEPTS = {
    "list": {"list", "Sequence", "Collection", "Iterable"},
    "tuple": {"tuple", "Sequence", "Collection", "Iterable"},
    "set": {"set", "Collection", "Iterable"},
    "frozenset": {"frozenset", "Collection", "Iterable"},
    "Sequence": {"Sequence", "Collection", "Iterable"},
    "Collection": {"Collection", "Iterable"},
    "Iterable": {"Iterable"},
    "Iterator": {"Iterator", "Iterable"},
    "dict": {"dict", "Mapping"},
    "Mapping": {"Mapping"},
    "AsyncIterator": {"AsyncIterator"},
}
_UNKNOWN = frozenset({"Any", "TypeVar", "Unknown"})
_TRANSPARENT = {"Annotated", "Final", "ClassVar", "Required", "NotRequired", "ReadOnly", "Awaitable", "Optional"}


@dataclass(frozen=True)
class TypeRef:
    """A normalised type annotation.

    ``name`` is the head of the type (``"DataFrame"``, ``"list"``, ``"Union"``, ...).
    ``qualname`` is the fully qualified name when the head resolves to a class, either
    inside the analysed SDK (``sdk=True``) or in another library.
    """

    name: str
    args: tuple[TypeRef, ...] = ()
    qualname: str | None = None
    sdk: bool = False

    # ------------------------------------------------------------------ predicates
    @property
    def is_unknown(self) -> bool:
        return self.name in _UNKNOWN

    @property
    def is_none(self) -> bool:
        return self.name == "None"

    @property
    def is_union(self) -> bool:
        return self.name == "Union"

    @property
    def is_primitive(self) -> bool:
        """True for builtin scalars and for containers/unions made only of them."""
        if self.is_union:
            return all(a.is_primitive for a in self.args)
        if self.name in PRIMITIVES or self.name == "Literal":
            return True
        if self.name in CONTAINERS:
            return all(a.is_primitive or a.is_unknown for a in self.args)
        return False

    @property
    def key(self) -> str:
        """Identity used for equality of class types (qualified when resolved)."""
        return self.qualname or self.name

    def alternatives(self) -> tuple[TypeRef, ...]:
        """Union members (flattened); a non-union type is its own single alternative."""
        if not self.is_union:
            return (self,)
        out: list[TypeRef] = []
        for arg in self.args:
            out.extend(arg.alternatives())
        return tuple(out)

    def class_refs(self) -> tuple[TypeRef, ...]:
        """All non-builtin class types mentioned anywhere in this type."""
        out: list[TypeRef] = []
        if self.is_union or self.name in CONTAINERS or self.name in {"Callable", "type"}:
            for arg in self.args:
                out.extend(arg.class_refs())
        elif not (self.name in PRIMITIVES or self.is_unknown or self.name in {"Literal", "Ellipsis"}):
            out.append(self)
        return tuple(out)

    # ------------------------------------------------------------------ (de)serialise
    def __str__(self) -> str:
        if self.is_union:
            return " | ".join(str(a) for a in self.args)
        if self.args:
            return f"{self.name}[{', '.join(str(a) for a in self.args)}]"
        return self.name

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        if self.args:
            out["args"] = [a.to_json() for a in self.args]
        if self.qualname:
            out["qualname"] = self.qualname
        if self.sdk:
            out["sdk"] = True
        return out

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> TypeRef | None:
        if data is None:
            return None
        return cls(
            name=data["name"],
            args=tuple(cls.from_json(a) for a in data.get("args", [])),  # type: ignore[misc]
            qualname=data.get("qualname"),
            sdk=bool(data.get("sdk", False)),
        )


ANY = TypeRef("Any")
NONE = TypeRef("None")

# Resolver: maps a dotted name as written in the source to ``(qualified name, is_sdk_class)``,
# or directly to a TypeRef (type aliases, bound type variables), or to None when unknown.
Resolver = Callable[[str], "tuple[str, bool] | TypeRef | None"]


def make_union(members: Iterable[TypeRef]) -> TypeRef:
    flat: list[TypeRef] = []
    seen: set[tuple[str, tuple[TypeRef, ...]]] = set()
    for member in members:
        for alt in member.alternatives():
            ident = (alt.key, alt.args)
            if ident not in seen:
                seen.add(ident)
                flat.append(alt)
    if not flat:
        return ANY
    if len(flat) == 1:
        return flat[0]
    return TypeRef("Union", tuple(flat))


def _dotted(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def parse_annotation(
    node: ast.expr | str | None,
    resolver: Resolver | None = None,
    self_type: TypeRef | None = None,
) -> TypeRef | None:
    """Parse an annotation AST node (or source string) into a :class:`TypeRef`.

    Returns ``None`` for a missing annotation.  Unparseable annotations become ``Any``.
    """
    if node is None:
        return None
    if isinstance(node, str):
        try:
            node = ast.parse(node.strip(), mode="eval").body
        except SyntaxError:
            return ANY
    return _parse(node, resolver, self_type, depth=0)


def _parse(node: ast.expr, resolver: Resolver | None, self_type: TypeRef | None, depth: int) -> TypeRef:
    if depth > 12:
        return ANY
    if isinstance(node, ast.Constant):
        if node.value is None:
            return NONE
        if isinstance(node.value, str):  # forward reference
            try:
                inner = ast.parse(node.value.strip(), mode="eval").body
            except SyntaxError:
                return ANY
            return _parse(inner, resolver, self_type, depth + 1)
        if node.value is Ellipsis:
            return TypeRef("Ellipsis")
        return TypeRef("Literal")
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return make_union(
            [_parse(node.left, resolver, self_type, depth + 1), _parse(node.right, resolver, self_type, depth + 1)]
        )
    if isinstance(node, (ast.Name, ast.Attribute)):
        dotted = _dotted(node)
        if dotted is None:
            return ANY
        return _named(dotted, (), resolver, self_type)
    if isinstance(node, ast.Subscript):
        head = _dotted(node.value)
        if head is None:
            return ANY
        short = head.rsplit(".", 1)[-1]
        slice_node = node.slice
        elems = list(slice_node.elts) if isinstance(slice_node, ast.Tuple) else [slice_node]
        if short == "Literal":
            return TypeRef("Literal")
        if short in _TRANSPARENT:
            inner = _parse(elems[0], resolver, self_type, depth + 1)
            if short == "Optional":
                return make_union([inner, NONE])
            return inner
        if short == "Union":
            return make_union([_parse(e, resolver, self_type, depth + 1) for e in elems])
        if short == "Callable":
            return TypeRef("Callable")
        args = tuple(
            _parse(e, resolver, self_type, depth + 1) for e in elems if not isinstance(e, (ast.List, ast.Starred))
        )
        return _named(head, args, resolver, self_type)
    if isinstance(node, ast.List):  # e.g. the argument list of Callable
        return ANY
    return ANY


def _named(dotted: str, args: tuple[TypeRef, ...], resolver: Resolver | None, self_type: TypeRef | None) -> TypeRef:
    short = dotted.rsplit(".", 1)[-1]
    if short == "Self" and self_type is not None:
        return self_type
    if short == "Any":
        return ANY
    canonical = _ALIASES.get(short, short)
    if canonical in PRIMITIVES or canonical in CONTAINERS or canonical in {"type", "Callable", "Awaitable"}:
        if resolver is not None and canonical not in PRIMITIVES:
            resolved = resolver(dotted)
            if isinstance(resolved, tuple) and resolved[1]:
                # An SDK class that shadows a builtin-looking name (e.g. ``Sequence``).
                return TypeRef(short, args, qualname=resolved[0], sdk=True)
        if canonical == "Awaitable" and args:
            return args[-1]
        return TypeRef(canonical, args)
    if resolver is not None:
        resolved = resolver(dotted)
        if isinstance(resolved, TypeRef):
            return resolved
        if resolved is not None:
            qualname, is_sdk = resolved
            return TypeRef(qualname.rsplit(".", 1)[-1], args, qualname=qualname, sdk=is_sdk)
    return TypeRef(short, args, qualname=dotted if "." in dotted else None)


# --------------------------------------------------------------------------------------
# Compatibility
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CompatWeights:
    """Scores assigned to the different ways a value of one type can flow into another."""

    exact: float = 1.0
    subclass: float = 0.9
    union_member: float = 0.8  # destination is a union with several class members
    superclass: float = 0.3  # would need a downcast
    unknown: float = 0.1  # unannotated / Any on either side
    primitive: float = 0.0  # str -> str and similar flows carry no structural signal
    container_element: float = 0.0  # X -> list[X]


DEFAULT_WEIGHTS = CompatWeights()

# Subclass oracle: is_subclass(child_qualname, parent_qualname).
SubclassFn = Callable[[str, str], bool]


def type_compat(
    src: TypeRef | None,
    dst: TypeRef | None,
    is_subclass: SubclassFn | None = None,
    weights: CompatWeights = DEFAULT_WEIGHTS,
) -> float:
    """Score in ``[0, 1]`` for passing a value of type ``src`` where ``dst`` is expected."""
    if src is None or dst is None or src.is_unknown or dst.is_unknown:
        return weights.unknown
    if src.is_none:
        return 0.0
    if src.is_union:
        members = [m for m in src.alternatives() if not m.is_none]
        if not members:
            return 0.0
        return sum(type_compat(m, dst, is_subclass, weights) for m in members) / len(members)
    if dst.is_union:
        members = [m for m in dst.alternatives() if not m.is_none]
        if not members:
            return 0.0
        best = max(type_compat(src, m, is_subclass, weights) for m in members)
        class_members = [m for m in members if not m.is_primitive]
        if len(class_members) > 1 and best >= weights.union_member:
            return min(best, weights.union_member)
        return best
    if dst.name == "object":
        return weights.unknown
    src_prim = src.name in PRIMITIVES or src.name == "Literal"
    dst_prim = dst.name in PRIMITIVES or dst.name == "Literal"
    if src_prim or dst_prim:
        if src_prim and dst_prim and (src.name == dst.name or (src.name, dst.name) in {("int", "float"), ("bool", "int")}):
            return weights.primitive
        return 0.0
    src_cont = src.name in CONTAINERS and not src.sdk
    dst_cont = dst.name in CONTAINERS and not dst.sdk
    if src_cont or dst_cont:
        if src_cont and dst_cont:
            if dst.name not in _CONTAINER_ACCEPTS.get(src.name, {src.name}):
                return 0.0
            return _container_args_compat(src, dst, is_subclass, weights)
        if dst_cont and dst.name in _SEQUENCE_LIKE and dst.args:
            if type_compat(src, dst.args[0], is_subclass, weights) >= weights.subclass:
                return weights.container_element
        return 0.0
    if src.key == dst.key or (src.qualname is None or dst.qualname is None) and src.name == dst.name:
        if src.args and dst.args and len(src.args) == len(dst.args):
            inner = min(
                (type_compat(a, b, is_subclass, weights) for a, b in zip(src.args, dst.args)),
                default=weights.exact,
            )
            return weights.exact if inner >= weights.subclass else max(weights.union_member, inner)
        return weights.exact
    if is_subclass is not None and src.qualname and dst.qualname:
        if is_subclass(src.qualname, dst.qualname):
            return weights.subclass
        if is_subclass(dst.qualname, src.qualname):
            return weights.superclass
    return 0.0


def _container_args_compat(src: TypeRef, dst: TypeRef, is_subclass: SubclassFn | None, weights: CompatWeights) -> float:
    """Compatibility of two builtin containers whose heads are already known to match."""
    if not dst.args or not src.args:
        # Bare ``list`` flowing into ``list[X]`` (or the reverse): only meaningful when a
        # class type is involved on the annotated side.
        annotated = dst.args or src.args
        if any(a.class_refs() for a in annotated):
            return weights.unknown
        return weights.primitive
    if src.name in _MAPPING_LIKE:
        pairs = list(zip(src.args[:2], dst.args[:2]))
    elif src.name == "tuple" and dst.name == "tuple":
        if len(src.args) != len(dst.args):
            return 0.0
        pairs = list(zip(src.args, dst.args))
    else:
        src_elem = make_union([a for a in src.args if a.name != "Ellipsis"])
        dst_elem = make_union([a for a in dst.args if a.name != "Ellipsis"])
        pairs = [(src_elem, dst_elem)]
    if all(a.is_primitive and b.is_primitive for a, b in pairs):
        return weights.primitive
    score = 1.0
    for a, b in pairs:
        if a.is_primitive and b.is_primitive:
            continue
        score = min(score, type_compat(a, b, is_subclass, weights))
    return score
