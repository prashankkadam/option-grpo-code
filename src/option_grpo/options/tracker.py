"""Initiation sets at decode time.

The initiation set of an option is ``I_k = {s : type(last_value(s)) ∈ T_in}``.  While
decoding, the type of the last value is tracked symbolically: after an option
terminates, the last value has the option's output type ``T_out``.  Options whose first
function needs no SDK object (*source* options) can be initiated in any state.

The successor relation used here is the composability graph ``G_opt``
(``o_i ≺ o_j`` iff ``T_out(o_i) ∩ T_in(o_j) ≠ ∅``) restricted to the options of the
current action space.
"""

from __future__ import annotations

from option_grpo.options.option import CodeOption, OptionVocabulary
from option_grpo.sdk.introspect import SDKIndex
from option_grpo.sdk.types import TypeRef, type_compat


class InitiationTable:
    """Precomputed initiation masks for one action space ``A_level``."""

    def __init__(
        self,
        vocab: OptionVocabulary,
        index: SDKIndex,
        level: int,
        mode: str = "last_value",
        min_compat: float = 0.5,
        allowed: set[int] | None = None,
    ) -> None:
        if mode not in {"last_value", "scope", "any"}:
            raise ValueError(f"unknown initiation mode: {mode}")
        self.vocab = vocab
        self.index = index
        self.level = level
        self.mode = mode
        self.min_compat = min_compat
        available = vocab.action_space(level)
        if allowed is not None:
            available = [o for o in available if o.index in allowed]
        self.available: tuple[int, ...] = tuple(o.index for o in available)
        self._available_set = frozenset(self.available)
        self.sources: tuple[int, ...] = tuple(o.index for o in available if o.is_source)
        self._successors: dict[int, tuple[int, ...]] = {}
        self._by_type: dict[str, tuple[int, ...]] = {}
        for option in available:
            if vocab.successors:
                nxt = [i for i in vocab.successors.get(option.index, []) if i in self._available_set]
            else:
                nxt = [o.index for o in available if o.in_types and vocab.composable(option, o, index, min_compat)]
            self._successors[option.index] = tuple(sorted(set(nxt) | set(self.sources)))

    # ------------------------------------------------------------------ queries
    def initial(self, value_types: list[TypeRef] | None = None) -> tuple[int, ...]:
        """Options initiable before any option ran, given the types of the setup values."""
        if self.mode == "any":
            return self.available
        allowed = set(self.sources)
        for vtype in value_types or []:
            allowed.update(self.accepting(vtype))
        return tuple(sorted(allowed))

    def accepting(self, vtype: TypeRef) -> tuple[int, ...]:
        """Available options whose ``T_in`` accepts a value of type ``vtype``."""
        key = f"{vtype.key}[{','.join(a.key for a in vtype.args)}]"
        cached = self._by_type.get(key)
        if cached is None:
            hits = []
            for idx in self.available:
                option = self.vocab[idx]
                if any(type_compat(vtype, t, self.index.is_subclass) >= self.min_compat for t in option.in_types):
                    hits.append(idx)
            cached = self._by_type[key] = tuple(hits)
        return cached

    def after(self, option_index: int) -> tuple[int, ...]:
        """Options initiable right after ``option_index`` terminated."""
        if self.mode == "any":
            return self.available
        return self._successors.get(option_index, self.sources)


class InitiationState:
    """Per-sample tracker of which options may be initiated next."""

    __slots__ = ("table", "allowed", "_scope")

    def __init__(self, table: InitiationTable, value_types: list[TypeRef] | None = None) -> None:
        self.table = table
        self.allowed: tuple[int, ...] = table.initial(value_types)
        self._scope: set[int] = set(self.allowed)

    def on_terminate(self, option: CodeOption, completed: bool) -> None:
        """Update the initiation set after an option ended.

        An aborted option produced no value of its output type, so the set is unchanged.
        """
        if not completed:
            return
        nxt = self.table.after(option.index)
        if self.table.mode == "scope":
            self._scope.update(nxt)
            self.allowed = tuple(sorted(self._scope))
        else:
            self.allowed = nxt
