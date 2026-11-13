"""Property-based test generation from type annotations (Algorithm 3).

For a function ``f`` with input types ``t_1..t_k`` and return type ``t_ret`` the suite
``P_f`` contains

* a **type preservation** test: every value ``f`` returns is an instance of ``t_ret``;
* **shape consistency** tests when ``t_ret`` has a length or a shape;
* a **determinism** test: re-running the program yields the same observable output;
* **edge-case** tests: the program is re-run on empty, single-element and boundary
  inputs produced by the input synthesiser;
* **invariant** tests: every ``assert`` over the parameters found in the source of
  ``f`` must hold for the arguments the program passes.

The tests are data (JSON) interpreted by the execution harness: they are evaluated on
the calls a candidate program actually makes (observed with a profiler hook) and on the
program's output, so no reference solution or hand-written test is involved.
"""

from __future__ import annotations

from typing import Any

from option_grpo.sdk.introspect import FunctionInfo, SDKIndex
from option_grpo.sdk.types import TypeRef

# Verbs whose result cannot be larger than their input / keeps the size of the input.
REDUCING = frozenset({
    "filter", "where", "limit", "head", "tail", "take", "sample", "distinct", "unique", "dedup", "deduplicate",
    "drop_duplicates", "dropna", "drop_nulls", "drop_nans", "exclude", "slice", "top", "top_k", "first", "search",
    "query", "find", "match",
})
PRESERVING = frozenset({
    "sort", "sort_by", "order_by", "orderby", "select", "with_column", "with_columns", "rename", "cast", "map",
    "alias", "reverse", "fill_null", "fillna", "fill_nan", "shuffle", "project",
})

_BUILTIN_RUNTIME = {
    "str": ["builtins.str"],
    "int": ["builtins.int"],
    "float": ["builtins.float", "builtins.int"],
    "bool": ["builtins.bool"],
    "bytes": ["builtins.bytes"],
    "bytearray": ["builtins.bytearray"],
    "complex": ["builtins.complex", "builtins.float", "builtins.int"],
    "list": ["builtins.list"],
    "dict": ["builtins.dict"],
    "tuple": ["builtins.tuple"],
    "set": ["builtins.set"],
    "frozenset": ["builtins.frozenset"],
    "Path": ["pathlib.Path", "pathlib.PurePath", "builtins.str"],
    "date": ["datetime.date"],
    "datetime": ["datetime.datetime"],
    "timedelta": ["datetime.timedelta"],
    "Decimal": ["decimal.Decimal"],
    "UUID": ["uuid.UUID"],
}
_STRUCTURAL = {
    "Sequence": "sequence",
    "Collection": "sized",
    "Iterable": "iterable",
    "Iterator": "iterable",
    "AsyncIterator": "async_iterable",
    "Mapping": "mapping",
    "Callable": "callable",
}
_SIZED_BUILTINS = {"list", "dict", "tuple", "set", "frozenset", "str", "bytes", "Sequence", "Collection", "Mapping"}


def runtime_accept(ret: TypeRef | None, index: SDKIndex) -> dict[str, Any]:
    """Acceptance spec for runtime values of a statically annotated type.

    ``names`` lists accepted class names (matched against the qualified and the bare
    names of the classes in the value's MRO), ``structural`` lists duck-typed protocols,
    ``none_ok`` tells whether ``None`` is acceptable and ``any`` disables the check.
    """
    spec: dict[str, Any] = {"names": [], "structural": [], "none_ok": False, "any": False}
    if ret is None or ret.is_unknown:
        spec["any"] = True
        return spec
    for alt in ret.alternatives():
        if alt.is_none:
            spec["none_ok"] = True
        elif alt.is_unknown or alt.name in {"object", "Literal", "type"}:
            spec["any"] = True
        elif alt.name in _STRUCTURAL and not alt.sdk:
            spec["structural"].append(_STRUCTURAL[alt.name])
        elif alt.name in _BUILTIN_RUNTIME and not alt.sdk:
            spec["names"].extend(_BUILTIN_RUNTIME[alt.name])
        else:
            if alt.qualname:
                spec["names"].append(alt.qualname)
                cls = index.classes.get(alt.qualname)
                if cls is not None and cls.public_path:
                    spec["names"].append(cls.public_path)
            spec["names"].append(alt.name)
    spec["names"] = sorted(set(spec["names"]))
    spec["structural"] = sorted(set(spec["structural"]))
    return spec


def has_size(ret: TypeRef | None, index: SDKIndex) -> bool:
    """Whether ``t_ret`` has a length or a shape (Algorithm 3, line 4)."""
    if ret is None:
        return False
    for alt in ret.alternatives():
        if alt.name in _SIZED_BUILTINS and not alt.sdk:
            return True
        if alt.sdk and alt.qualname:
            for ancestor in index.mro(alt.qualname):
                cls = index.classes.get(ancestor)
                if cls is None:
                    continue
                if f"{ancestor}.__len__" in index.functions or {"shape", "num_rows"} & set(cls.attributes):
                    return True
    return False


def shape_relation(fn: FunctionInfo) -> str:
    if fn.name in REDUCING:
        return "<="
    if fn.name in PRESERVING:
        return "=="
    return "defined"


def function_locator(fn: FunctionInfo) -> dict[str, Any]:
    """How the harness finds the function object at run time."""
    attrs: list[str] = []
    if fn.owner is not None:
        attrs.extend(fn.owner[len(fn.module) + 1 :].split("."))
        attrs.append("__init__" if fn.kind == "constructor" else fn.name)
    else:
        attrs.append(fn.name)
    return {"id": fn.id, "module": fn.module, "attrs": attrs, "name": fn.name, "kind": fn.kind}


def function_tests(fn: FunctionInfo, index: SDKIndex) -> list[dict[str, Any]]:
    """The function-level part of ``P_f``: type, shape and invariant tests."""
    tests: list[dict[str, Any]] = []
    accept = runtime_accept(fn.returns, index)
    if not accept["any"]:
        tests.append({"kind": "call_type", "function": fn.id, "accept": accept})
    if has_size(fn.returns, index):
        tests.append({"kind": "shape", "function": fn.id, "relation": shape_relation(fn)})
    for expr in fn.asserts:
        tests.append({"kind": "invariant", "function": fn.id, "expr": expr})
    return tests


def task_tests(
    functions: list[FunctionInfo],
    index: SDKIndex,
    variant_names: list[str],
    max_per_function: int = 15,
) -> list[dict[str, Any]]:
    """Property-test suite of a task generated from the path ``functions``."""
    tests: list[dict[str, Any]] = []
    seen: set[str] = set()
    output_accept = runtime_accept(functions[-1].returns, index)
    tests.append({"kind": "type", "accept": output_accept})
    if has_size(functions[-1].returns, index):
        tests.append({"kind": "output_shape"})
    tests.append({"kind": "determinism"})
    for name in variant_names:
        tests.append({"kind": "edge", "variant": name, "expect": "any"})
    for fn in functions:
        if fn.id in seen:
            continue
        seen.add(fn.id)
        budget = max(max_per_function - (3 + len(variant_names)), 1)
        tests.extend(function_tests(fn, index)[:budget])
    for number, test in enumerate(tests):
        test["id"] = f"{test['kind']}-{number}"
    return tests


def count_function_tests(fn: FunctionInfo, index: SDKIndex, n_variants: int = 3) -> int:
    """Size of ``P_f`` for a single-function task (used for benchmark statistics)."""
    return len(function_tests(fn, index)) + 2 + n_variants + (1 if has_size(fn.returns, index) else 0)
