"""Code options and the option vocabulary.

A code option is the 4-tuple ``o_k = (I_k, F_k, beta_k, pi_k)`` of Definition 1:

* ``I_k`` -- the initiation set, represented by the input types ``T_in`` of the first
  function of the option (:attr:`CodeOption.in_types`);
* ``F_k`` -- the allowed SDK function set, the centroid path of the option
  (:attr:`CodeOption.path`) plus the argument constructors that may appear inside the
  calls (:attr:`CodeOption.helpers`);
* ``beta_k`` -- the termination predicate, true once the last function of the path has
  returned a value of type ``T_out`` (:attr:`CodeOption.out_types`);
* ``pi_k`` -- the intra-option policy, i.e. the shared network ``pi_theta`` with every
  token that would resolve to a call outside ``F_k`` masked out
  (see :mod:`option_grpo.options.constraints`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from option_grpo.sdk.introspect import FunctionInfo, SDKIndex
from option_grpo.sdk.types import TypeRef, type_compat
from option_grpo.utils import read_json, stable_hash, write_json

CANDIDATE = "candidate"
CERTIFIED = "certified"
REJECTED = "rejected"


@dataclass
class CodeOption:
    index: int  # position in the vocabulary; stable across phases
    level: int  # 0 = atomic call, 1 = composed pattern
    key: str  # unique, human-readable identifier
    path: tuple[str, ...]  # function ids of the centroid path (F_k)
    call_names: tuple[str, ...]  # names of the path functions as written at call sites
    in_types: tuple[TypeRef, ...]  # T_in: class types consumed by the first function
    out_types: tuple[TypeRef, ...]  # T_out: type returned by the last function
    is_source: bool  # can start from primitives only (always initiable)
    helpers: frozenset[str] = frozenset()  # call names permitted inside argument lists
    weight: float = 1.0  # path weight, the product of type-compatibility scores
    members: list[tuple[str, ...]] = field(default_factory=list)  # paths of the cluster (Level 1)
    status: str = CANDIDATE
    stats: dict[str, Any] = field(default_factory=dict)  # certification statistics

    def __len__(self) -> int:
        return len(self.path)

    @property
    def certified(self) -> bool:
        return self.status == CERTIFIED

    @property
    def label(self) -> str:
        return " -> ".join(self.call_names)

    def member_patterns(self) -> set[tuple[str, ...]]:
        """Call-name sequences that count as an invocation of this option."""
        return {self.call_names}

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "level": self.level,
            "key": self.key,
            "path": list(self.path),
            "call_names": list(self.call_names),
            "in_types": [t.to_json() for t in self.in_types],
            "out_types": [t.to_json() for t in self.out_types],
            "is_source": self.is_source,
            "helpers": sorted(self.helpers),
            "weight": self.weight,
            "members": [list(m) for m in self.members],
            "status": self.status,
            "stats": self.stats,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> CodeOption:
        return cls(
            index=d["index"],
            level=d["level"],
            key=d["key"],
            path=tuple(d["path"]),
            call_names=tuple(d["call_names"]),
            in_types=tuple(TypeRef.from_json(t) for t in d["in_types"]),  # type: ignore[misc]
            out_types=tuple(TypeRef.from_json(t) for t in d["out_types"]),  # type: ignore[misc]
            is_source=d["is_source"],
            helpers=frozenset(d.get("helpers", [])),
            weight=d.get("weight", 1.0),
            members=[tuple(m) for m in d.get("members", [])],
            status=d.get("status", CANDIDATE),
            stats=dict(d.get("stats", {})),
        )


def option_io_types(first: FunctionInfo, last: FunctionInfo) -> tuple[tuple[TypeRef, ...], tuple[TypeRef, ...], bool]:
    """``(T_in, T_out, is_source)`` for an option whose path starts at ``first`` and ends at ``last``."""
    in_types: list[TypeRef] = []
    needs_object = first.receiver is not None
    if first.receiver is not None:
        in_types.append(first.receiver)
    for param in first.params:
        if param.type is None:
            continue
        for alt in param.type.alternatives():
            for ref in alt.class_refs() if not alt.is_primitive else ():
                if ref not in in_types:
                    in_types.append(ref)
        if param.required and param.kind == "positional" and not param.type.is_primitive and param.type.class_refs():
            if not any(a.is_primitive for a in param.type.alternatives()):
                needs_object = True
    out_types: tuple[TypeRef, ...] = ()
    if last.returns is not None and not last.returns.is_none:
        out_types = tuple(a for a in last.returns.alternatives() if not a.is_none)
    return tuple(in_types), out_types, not needs_object


def make_option_key(level: int, functions: Iterable[FunctionInfo]) -> str:
    functions = list(functions)
    if level == 0:
        return f"L0:{functions[0].public_path or functions[0].id}"
    names = ">".join(fn.name for fn in functions)
    return f"L1:{names}#{stable_hash(*[fn.id for fn in functions], digits=6)}"


class OptionVocabulary:
    """All options of one SDK, with the nested action spaces ``A_0 ⊂ A_1 ⊂ A_2``.

    ``A_0`` contains no options (token-level actions only), ``A_1`` adds the certified
    Level-0 options and ``A_2`` adds the certified Level-1 options.
    """

    def __init__(self, sdk: str, options: list[CodeOption] | None = None) -> None:
        self.sdk = sdk
        self.options: list[CodeOption] = []
        self._by_key: dict[str, CodeOption] = {}
        self.successors: dict[int, list[int]] = {}  # the composability graph G_opt
        for option in options or []:
            self._append(option)

    # ------------------------------------------------------------------ construction
    def _append(self, option: CodeOption) -> None:
        if option.key in self._by_key:
            raise ValueError(f"duplicate option key: {option.key}")
        if option.index != len(self.options):
            raise ValueError(f"option index {option.index} does not match its position {len(self.options)}")
        self.options.append(option)
        self._by_key[option.key] = option

    def add(self, option: CodeOption) -> CodeOption:
        option.index = len(self.options)
        self._append(option)
        return option

    def __len__(self) -> int:
        return len(self.options)

    def __getitem__(self, index: int) -> CodeOption:
        return self.options[index]

    def get(self, key: str) -> CodeOption | None:
        return self._by_key.get(key)

    # ------------------------------------------------------------------ views
    def level(self, level: int, status: str | None = None) -> list[CodeOption]:
        return [o for o in self.options if o.level == level and (status is None or o.status == status)]

    def certified(self, level: int | None = None) -> list[CodeOption]:
        return [o for o in self.options if o.certified and (level is None or o.level == level)]

    def action_space(self, level: int) -> list[CodeOption]:
        """Options available in ``A_level``: certified options of every lower level."""
        return [o for o in self.options if o.certified and o.level < level]

    def patterns(self, levels: Iterable[int], certified_only: bool = True) -> dict[tuple[str, ...], CodeOption]:
        """Map call-name sequences to options (the higher-weight option wins a collision)."""
        wanted = set(levels)
        table: dict[tuple[str, ...], CodeOption] = {}
        for option in self.options:
            if option.level not in wanted or (certified_only and not option.certified):
                continue
            current = table.get(option.call_names)
            if current is None or option.weight > current.weight:
                table[option.call_names] = option
        return table

    def by_function(self, level: int = 0) -> dict[str, CodeOption]:
        return {o.path[0]: o for o in self.options if o.level == level}

    # ------------------------------------------------------------------ G_opt
    def composable(self, first: CodeOption, second: CodeOption, index: SDKIndex, min_compat: float = 0.5) -> bool:
        """``first ≺ second``: some output type of ``first`` is accepted by ``second``."""
        for out_type in first.out_types:
            for in_type in second.in_types:
                if type_compat(out_type, in_type, index.is_subclass) >= min_compat:
                    return True
        return False

    def build_composability(self, index: SDKIndex, min_compat: float = 0.5) -> None:
        """(Re)compute the successor lists of ``G_opt`` over all options."""
        self.successors = {}
        for first in self.options:
            self.successors[first.index] = [
                second.index
                for second in self.options
                if second.in_types and self.composable(first, second, index, min_compat)
            ]

    # ------------------------------------------------------------------ statistics
    def stats(self) -> dict[str, Any]:
        level1 = self.level(1)
        certified1 = [o for o in level1 if o.certified]
        pool = certified1 or level1
        return {
            "sdk": self.sdk,
            "level0": len(self.level(0)),
            "level0_certified": len(self.certified(0)),
            "level1_candidates": len(level1),
            "level1_certified": len(certified1),
            "avg_length": round(sum(len(o) for o in pool) / len(pool), 2) if pool else 0.0,
        }

    # ------------------------------------------------------------------ persistence
    def to_json(self) -> dict[str, Any]:
        return {
            "sdk": self.sdk,
            "stats": self.stats(),
            "options": [o.to_json() for o in self.options],
            "successors": {str(k): v for k, v in self.successors.items()},
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> OptionVocabulary:
        vocab = cls(d["sdk"], [CodeOption.from_json(o) for o in d["options"]])
        vocab.successors = {int(k): list(v) for k, v in d.get("successors", {}).items()}
        return vocab

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_json(), indent=1)

    @classmethod
    def load(cls, path: str | Path) -> OptionVocabulary:
        return cls.from_json(read_json(path))
