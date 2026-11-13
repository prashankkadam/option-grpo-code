"""Input synthesis from type annotations ("type constructors from ``G_api``").

Given a function of the SDK, the synthesiser produces minimal valid arguments:

* primitive parameters receive literals chosen from the annotation and, where the
  annotation is too coarse (``str``), from the parameter and function names (a path
  parameter of ``read_csv`` becomes a CSV fixture, a ``by`` parameter a column name);
* parameters typed with an SDK class are built by calling a *type constructor* of that
  class found in ``G_api`` (a function returning the class whose own inputs can be
  synthesised recursively).

The same machinery yields the edge-case variants (empty, single element, boundary
values) used by the property tests of Algorithm 3.
"""

from __future__ import annotations

import pprint
from dataclasses import dataclass, field

from option_grpo.graphs.build import SDKGraphs
from option_grpo.sdk.introspect import FunctionInfo, ParamInfo
from option_grpo.sdk.types import TypeRef
from option_grpo.tasks.fixtures import COLUMNS, EXTENSIONS, ROWS, Fixture
from option_grpo.utils import first_sentence, split_identifier


class SynthError(Exception):
    """Raised when no valid input can be synthesised for a parameter."""


_PATH_WORDS = {"path", "file", "filename", "filepath", "fname", "fpath", "source", "src", "document", "input"}
_DIR_WORDS = {"uri", "url", "dir", "directory", "folder", "root", "db", "database", "location", "workdir", "store"}
_COLUMN_WORDS = {"col", "column", "columns", "cols", "by", "key", "keys", "on", "field", "fields", "subset", "group"}
_COLUMN_FUNCTIONS = {"col", "column", "field", "groupby", "group_by", "sort", "sort_by", "order_by", "select", "drop"}
_COUNT_WORDS = {"n", "k", "limit", "num", "count", "size", "rows", "top", "batch", "max", "take", "head"}
_INDEX_WORDS = {"index", "idx", "offset", "start", "skip", "position", "pos"}
_LOW_WORDS = {"low", "lower", "min", "minimum", "start", "begin", "left"}
_HIGH_WORDS = {"high", "upper", "max", "maximum", "end", "stop", "right"}
_SEQUENCE_HEADS = {"list", "tuple", "set", "frozenset", "Sequence", "Collection", "Iterable", "Iterator"}
_MAPPING_HEADS = {"dict", "Mapping"}


@dataclass
class Value:
    """A synthesised argument."""

    code: str  # Python source of the value
    type: TypeRef | None = None
    n_calls: int = 0  # SDK calls contained in ``code``
    fixtures: list[Fixture] = field(default_factory=list)
    imports: set[str] = field(default_factory=set)
    variants: dict[str, str] = field(default_factory=dict)  # edge-case name -> alternative source
    description: str = ""  # natural-language rendering used in task statements
    primitive: bool = True


def describe_function(fn: FunctionInfo) -> str:
    """Short imperative phrase for a function, taken from its docstring when available."""
    if fn.kind == "constructor":
        return "create a " + " ".join(split_identifier(fn.name)) + " object"
    phrase = first_sentence(fn.doc)
    if not phrase or len(phrase) > 160:
        return " ".join(split_identifier(fn.name))
    if len(phrase) > 1 and phrase[0].isupper() and not phrase[1].isupper():
        phrase = phrase[0].lower() + phrase[1:]
    return phrase


class Synthesizer:
    """Builds argument expressions for SDK calls."""

    def __init__(
        self,
        graphs: SDKGraphs,
        hints: dict[str, str] | None = None,
        min_compat: float = 0.5,
        max_depth: int = 3,
    ) -> None:
        self.graphs = graphs
        self.index = graphs.index
        self.hints = dict(hints or {})
        self.min_compat = min_compat
        self.max_depth = max_depth
        self._producer_cache: dict[str, list[FunctionInfo]] = {}

    # ------------------------------------------------------------------ call syntax
    def callee(self, fn: FunctionInfo, receiver: str | None = None) -> str:
        """Source of the callable expression for ``fn``."""
        if fn.kind in {"method", "property"}:
            if receiver is None:
                raise SynthError(f"{fn.id} needs a receiver")
            return f"{receiver}.{fn.name}"
        if fn.public_path is None:
            raise SynthError(f"{fn.id} is not public")
        return fn.public_path

    def render_call(self, fn: FunctionInfo, receiver: str | None, args: dict[str, str]) -> str:
        """Source of a call to ``fn`` given the source of each supplied argument."""
        callee = self.callee(fn, receiver)
        if fn.kind == "property":
            return callee
        rendered: list[str] = []
        keyword_only = False
        for param in fn.params:
            if param.name not in args:
                if param.kind == "positional" and not param.has_default:
                    raise SynthError(f"missing required argument {param.name!r} of {fn.id}")
                if param.kind == "positional":
                    keyword_only = True  # skipped a positional parameter: the rest go by keyword
                continue
            code = args[param.name]
            if param.kind == "vararg":
                rendered.append(code)
            elif param.kind == "kwarg":
                rendered.append(f"**{code}")
            elif param.kind == "keyword" or keyword_only:
                rendered.append(f"{param.name}={code}")
            else:
                rendered.append(code)
        call = f"{callee}({', '.join(rendered)})"
        return f"asyncio.run({call})" if fn.is_async else call

    # ------------------------------------------------------------------ primitives
    def _words(self, fn: FunctionInfo, param: ParamInfo) -> tuple[set[str], set[str]]:
        return set(split_identifier(param.name)), set(split_identifier(fn.name))

    def _extension(self, fn: FunctionInfo, param: ParamInfo) -> str | None:
        for word in split_identifier(param.name) + split_identifier(fn.name):
            if word in EXTENSIONS:
                return word
        return None

    def _column(self, fn: FunctionInfo, param: ParamInfo, position: int) -> str:
        words, _ = self._words(fn, param)
        if words & {"key", "keys", "on", "group"} or fn.name in {"groupby", "group_by"}:
            preferred = ["department", "name"]
        else:
            preferred = ["age", "salary", "score", "id"]
        return preferred[position % len(preferred)]

    def _string(self, fn: FunctionInfo, param: ParamInfo, position: int) -> Value:
        pwords, fwords = self._words(fn, param)
        extension = self._extension(fn, param)
        if extension is not None and (pwords & (_PATH_WORDS | _DIR_WORDS) or extension in pwords or len(fn.required_params) == 1):
            kind = EXTENSIONS[extension]
            suffix = "txt" if kind == "txt" else extension
            fixture = Fixture(f"data.{suffix}", kind)
            return Value(repr(fixture.path), TypeRef("str"), fixtures=[fixture], description=f"the file `{fixture.path}`")
        if pwords & _PATH_WORDS and not pwords & _DIR_WORDS:
            fixture = Fixture("data.txt", "txt")
            return Value(repr(fixture.path), TypeRef("str"), fixtures=[fixture], description=f"the file `{fixture.path}`")
        if pwords & _DIR_WORDS:
            fixture = Fixture("workspace", "dir")
            return Value(repr(fixture.path), TypeRef("str"), fixtures=[fixture], description=f"the location `{fixture.path}`")
        if pwords & _COLUMN_WORDS or (fn.name in _COLUMN_FUNCTIONS and pwords & {"name", "names"}):
            column = self._column(fn, param, position)
            return Value(repr(column), TypeRef("str"), description=f"the column `{column}`")
        if pwords & {"name", "alias", "label", "title", "table", "id", "tag"}:
            text = "_".join(sorted(fwords - {"create", "open", "get", "set", "add", "make", "new"}))[:24] or "sample"
            text = "items" if fwords & {"table"} else ("value" if fwords & {"alias", "column"} else text)
            return Value(repr(text), TypeRef("str"), description=f"the name {text!r}")
        return Value(repr("sample"), TypeRef("str"), description="the text 'sample'")

    def _number(self, fn: FunctionInfo, param: ParamInfo, kind: str) -> Value:
        pwords, _ = self._words(fn, param)
        if kind == "int":
            if pwords & _INDEX_WORDS:
                return Value("1", TypeRef("int"), variants={"boundary": "0"})
            if pwords & _COUNT_WORDS:
                return Value("3", TypeRef("int"), variants={"boundary": "0", "single": "1"})
            return Value("40", TypeRef("int"), variants={"boundary": "0"})
        if pwords & _LOW_WORDS:
            return Value("30.0", TypeRef("float"), variants={"boundary": "0.0"})
        if pwords & _HIGH_WORDS:
            return Value("50.0", TypeRef("float"), variants={"boundary": "1e9"})
        return Value("40.0", TypeRef("float"), variants={"boundary": "0.0"})

    def _element(self, fn: FunctionInfo, param: ParamInfo, elem: TypeRef | None, position: int) -> list[str]:
        """Three sample elements for a container parameter."""
        pwords, _ = self._words(fn, param)
        if elem is None or elem.is_unknown:
            return ["1", "2", "3"]
        head = next((a for a in elem.alternatives() if not a.is_none), elem)
        if head.name in _MAPPING_HEADS:
            return [pprint.pformat(row, width=100, sort_dicts=False) for row in ROWS[:3]]
        if head.name == "str":
            if pwords & _COLUMN_WORDS:
                return [repr(c) for c in ("age", "salary", "score")]
            return [repr(s) for s in ("alpha", "beta", "gamma")]
        if head.name in {"int", "bool"}:
            return ["1", "2", "3"] if head.name == "int" else ["True", "False", "True"]
        if head.name == "float":
            return ["0.1", "0.2", "0.3"]
        if head.name in _SEQUENCE_HEADS:
            inner = self._element(fn, param, head.args[0] if head.args else None, position)
            return [f"[{', '.join(inner)}]"] * 2
        raise SynthError(f"cannot build elements of type {elem}")

    def primitive(self, fn: FunctionInfo, param: ParamInfo, position: int = 0) -> Value:
        """Literal for a parameter whose type involves no SDK class."""
        for key in (f"{fn.name}.{param.name}", param.name):
            if key in self.hints:
                return Value(self.hints[key], param.type, description=f"`{self.hints[key]}`")
        ptype = param.type
        if ptype is None or ptype.is_unknown:
            pwords, _ = self._words(fn, param)
            if pwords & (_PATH_WORDS | _DIR_WORDS | _COLUMN_WORDS | {"name"}):
                return self._string(fn, param, position)
            return Value("40", ptype)
        alternatives = [a for a in ptype.alternatives() if not a.is_none]
        order = {"str": 0, "int": 1, "float": 2, "bool": 3}
        alternatives.sort(key=lambda a: order.get(a.name, 4 if a.name in _SEQUENCE_HEADS | _MAPPING_HEADS else 9))
        errors: list[str] = []
        for alt in alternatives:
            try:
                return self._primitive_alt(fn, param, alt, position)
            except SynthError as exc:
                errors.append(str(exc))
        raise SynthError(f"{fn.id}: no literal for parameter {param.name!r} ({'; '.join(errors) or ptype})")

    def _primitive_alt(self, fn: FunctionInfo, param: ParamInfo, alt: TypeRef, position: int) -> Value:
        name = alt.name
        if name == "str":
            return self._string(fn, param, position)
        if name == "Path":
            value = self._string(fn, param, position)
            return Value(f"Path({value.code})", alt, fixtures=value.fixtures, imports={"from pathlib import Path"}, description=value.description)
        if name in {"int", "float"}:
            return self._number(fn, param, name)
        if name == "bool":
            return Value("True", alt)
        if name in {"bytes", "bytearray"}:
            return Value("b'sample'", alt, variants={"empty": "b''"})
        if name in _SEQUENCE_HEADS:
            elem = alt.args[0] if alt.args else None
            items = self._element(fn, param, elem, position)
            if name == "tuple":
                code = f"({', '.join(items)})"
                single = f"({items[0]},)"
                empty = "()"
            elif name in {"set", "frozenset"}:
                code, single, empty = f"{{{', '.join(items)}}}", f"{{{items[0]}}}", "set()"
            else:
                joined = ",\n    ".join(items)
                code = f"[\n    {joined},\n]" if len(joined) > 80 else f"[{', '.join(items)}]"
                single, empty = f"[{items[0]}]", "[]"
            return Value(code, alt, variants={"empty": empty, "single": single})
        if name in _MAPPING_HEADS:
            value_type = alt.args[1] if len(alt.args) > 1 else None
            if value_type is not None and value_type.name in _SEQUENCE_HEADS:
                columns = {c: [row[c] for row in ROWS[:3]] for c in COLUMNS}
                return Value(
                    pprint.pformat(columns, width=100, sort_dicts=False),
                    alt,
                    variants={"empty": repr({c: [] for c in COLUMNS}), "single": repr({c: [ROWS[0][c]] for c in COLUMNS})},
                )
            return Value(pprint.pformat(ROWS[0], width=100, sort_dicts=False), alt, variants={"empty": "{}"})
        if name == "Callable":
            return Value("lambda value: value", alt)
        if name == "object":
            return Value("40", alt)
        raise SynthError(f"unsupported primitive type {alt}")

    # ------------------------------------------------------------------ SDK objects
    def is_object_type(self, ptype: TypeRef | None) -> bool:
        """True when the parameter needs an instance of a (non-builtin) class."""
        if ptype is None or ptype.is_unknown:
            return False
        alternatives = [a for a in ptype.alternatives() if not a.is_none]
        if not alternatives:
            return False
        return all(a.class_refs() and not a.is_primitive and a.name != "Callable" for a in alternatives)

    def producers(self, target: TypeRef) -> list[FunctionInfo]:
        key = str(target) + "|" + (target.qualname or "")
        cached = self._producer_cache.get(key)
        if cached is not None:
            return cached
        ranked: list[tuple[tuple, FunctionInfo]] = []
        for fn, score in self.graphs.producers(target, self.min_compat):
            if fn.kind == "property":
                continue
            object_inputs = sum(1 for _, t in fn.input_types() if self.is_object_type(t))
            opaque = sum(
                1 for p in fn.required_params
                if p.type is None or p.type.is_unknown or any(a.name == "Callable" for a in p.type.alternatives())
            )
            kind_rank = {"function": 0, "classmethod": 1, "staticmethod": 1, "constructor": 2, "method": 3}.get(fn.kind, 4)
            ranked.append(((opaque, object_inputs, kind_rank, -score, len(fn.required_params), fn.id), fn))
        ranked.sort(key=lambda item: item[0])
        result = [fn for _, fn in ranked]
        self._producer_cache[key] = result
        return result

    def build(self, target: TypeRef, depth: int = 0, stack: tuple[str, ...] = ()) -> Value:
        """Expression that constructs a value of the SDK type ``target``."""
        if depth >= self.max_depth:
            raise SynthError(f"construction of {target} exceeds depth {self.max_depth}")
        last_error: SynthError | None = None
        for fn in self.producers(target):
            if fn.id in stack:
                continue
            try:
                return self.call(fn, depth=depth, stack=stack + (fn.id,))
            except SynthError as exc:
                last_error = exc
        raise SynthError(f"no type constructor for {target}" + (f" ({last_error})" if last_error else ""))

    def argument(self, fn: FunctionInfo, param: ParamInfo, position: int, depth: int, stack: tuple[str, ...]) -> Value:
        """Value for one parameter: a literal, or a constructed SDK object."""
        if self.is_object_type(param.type):
            assert param.type is not None
            errors: list[str] = []
            for alt in param.type.alternatives():
                if alt.is_none:
                    continue
                try:
                    value = self.build(alt, depth + 1, stack)
                    value.primitive = False
                    return value
                except SynthError as exc:
                    errors.append(str(exc))
            raise SynthError("; ".join(errors))
        return self.primitive(fn, param, position)

    def call(
        self,
        fn: FunctionInfo,
        depth: int = 0,
        stack: tuple[str, ...] = (),
        supplied: dict[str, str] | None = None,
        receiver: str | None = None,
    ) -> Value:
        """A complete call to ``fn`` with every required input synthesised inline.

        ``supplied`` maps parameter names to already available sources (flow values);
        ``receiver`` supplies the object a method is called on.
        """
        supplied = dict(supplied or {})
        n_calls = 0 if fn.kind == "property" else 1
        fixtures: list[Fixture] = []
        imports: set[str] = set()
        details: list[str] = []
        receiver_code = receiver
        if fn.receiver is not None and receiver_code is None:
            built = self.build(fn.receiver, depth + 1, stack)
            receiver_code, n_calls = built.code, n_calls + built.n_calls
            fixtures += built.fixtures
            imports |= built.imports
            if built.description:
                details.append(built.description)
        args: dict[str, str] = {}
        position = 0
        for param in fn.params:
            if param.name in supplied:
                args[param.name] = supplied[param.name]
                continue
            if not param.required and param.kind != "vararg":
                continue
            if param.kind == "kwarg":
                continue
            if param.kind == "vararg" and param.type is None:
                continue
            value = self.argument(fn, param, position, depth, stack)
            position += 1
            args[param.name] = value.code
            n_calls += value.n_calls
            fixtures += value.fixtures
            imports |= value.imports
            if value.primitive:
                details.append(f"{param.name}={value.code}")
            elif value.description:
                details.append(f"{param.name}: {value.description}")
        if fn.is_async:
            imports.add("import asyncio")
        code = self.render_call(fn, receiver_code, args)
        phrase = describe_function(fn)
        description = f"{phrase} ({', '.join(details)})" if details else phrase
        return Value(code, fn.returns, n_calls, fixtures, imports, {}, description, primitive=False)
