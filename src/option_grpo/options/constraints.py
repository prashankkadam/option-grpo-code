"""Decode-time realisation of the intra-option policy ``pi_k``.

While an option is active, ``pi_k`` is the shared network ``pi_theta`` with the logits
of every token whose *next-call resolution* falls outside the option's function set
``F_k`` set to ``-inf``.  This module makes that statement operational:

1. :class:`OptionAutomaton` is a character-level scanner of the Python text emitted
   since the option started.  It tracks string/comment state, bracket nesting, the
   identifier being typed and how far along the option's call path the text is.
2. The rules (see :meth:`OptionAutomaton.feed`) are

   * on the main call chain (outside any argument list) an SDK call must be the *next*
     function of the option's path, and a name after a dot must be that function or an
     SDK namespace/attribute name;
   * inside argument lists, SDK calls are restricted to the option's helper set (type
     constructors of the parameters, e.g. ``col`` inside ``where``);
   * everything else (variables, literals, operators, builtins) is unconstrained.

3. The option terminates -- ``beta_k`` fires -- when the bracket of the last path
   call closes, i.e. when a value of type ``T_out`` has been produced.
4. :class:`ConstraintEngine` turns an automaton state into a token mask.  Tokens that
   contain none of the structural characters are decided in closed form from an index
   over the vocabulary; the remaining tokens are simulated through the automaton and
   the result is cached under a reduced state key.
"""

from __future__ import annotations

import builtins
import keyword
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from option_grpo.options.option import CodeOption
from option_grpo.sdk.introspect import SDKIndex

KEYWORDS = frozenset(keyword.kwlist) | {"match", "case"}
STRUCTURAL = frozenset(".()[]{}'\"#\\\n")
_WS = frozenset(" \t\r\n\f\v")

# Lexical modes.
_NORMAL, _STRING, _COMMENT, _OPENING = 0, 1, 2, 3
# Bracket frames.
_GROUP, _CALL, _PATH, _PATH_LAST = 0, 1, 2, 3
# Kind of the last significant token.
_SIG_OTHER, _SIG_IDENT, _SIG_CLOSE = 0, 1, 2


def _is_ident(ch: str) -> bool:
    return ch == "_" or ch.isalnum()


def _prefixes(names: Iterable[str]) -> frozenset[str]:
    out: set[str] = set()
    for name in names:
        for end in range(1, len(name) + 1):
            out.add(name[:end])
    return frozenset(out)


# --------------------------------------------------------------------------------------
# Static tables
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LanguageTables:
    """Name sets of one SDK used to decide what a call or attribute resolves to."""

    callable: frozenset[str]  # every public call name (functions, methods, classes)
    bare_callable: frozenset[str]  # callable without a receiver (functions, classes)
    namespace: frozenset[str]  # modules, classes and attributes (may follow a dot freely)
    attr_prefixes: frozenset[str]  # prefixes of callable ∪ namespace
    bare_prefixes: frozenset[str]  # prefixes of bare_callable ∪ keywords
    namespace_prefixes: frozenset[str]  # prefixes of namespace

    @classmethod
    def from_index(cls, index: SDKIndex) -> LanguageTables:
        callable_names = frozenset(index.callable_names())
        python_builtins = frozenset(dir(builtins))
        bare = frozenset(index.bare_callable_names()) - python_builtins
        namespace = frozenset(index.namespace_names())
        return cls(
            callable=callable_names,
            bare_callable=bare,
            namespace=namespace,
            attr_prefixes=_prefixes(callable_names | namespace),
            bare_prefixes=_prefixes(bare | KEYWORDS),
            namespace_prefixes=_prefixes(namespace),
        )


@dataclass(frozen=True)
class ConstraintSpec:
    """Per-option constants of the automaton."""

    option_index: int
    names: tuple[str, ...]  # call names of the path, in order
    helpers: frozenset[str]
    lang: LanguageTables
    strict: bool = True

    @classmethod
    def from_option(cls, option: CodeOption, lang: LanguageTables, strict: bool = True) -> ConstraintSpec:
        return cls(option.index, option.call_names, option.helpers, lang, strict)


# --------------------------------------------------------------------------------------
# Automaton
# --------------------------------------------------------------------------------------
class OptionAutomaton:
    """Scanner over the text generated while an option is active."""

    __slots__ = (
        "spec", "lex", "quote", "triple", "escape", "qrun", "run", "run_dot", "run_num",
        "pdot", "must_call", "sig", "ident", "ident_dot", "stack", "nargs", "progress",
        "terminated", "violated",
    )

    def __init__(self, spec: ConstraintSpec, pending_dot: bool = False) -> None:
        self.spec = spec
        self.lex = _NORMAL
        self.quote = ""
        self.triple = False
        self.escape = False
        self.qrun = 0
        self.run: str | None = None  # identifier / number currently being typed
        self.run_dot = False  # the run follows a dot (attribute access)
        self.run_num = False  # the run is a numeric literal
        self.pdot = pending_dot  # a dot was just written
        self.must_call = False  # the last name was the next path function: "(" must follow
        self.sig = _SIG_OTHER
        self.ident = ""
        self.ident_dot = False
        self.stack: list[int] = []
        self.nargs = 0  # number of enclosing call brackets (argument depth)
        self.progress = 0  # number of path calls already opened
        self.terminated = False
        self.violated = False

    def copy(self) -> OptionAutomaton:
        other = OptionAutomaton.__new__(OptionAutomaton)
        other.spec = self.spec
        other.lex = self.lex
        other.quote = self.quote
        other.triple = self.triple
        other.escape = self.escape
        other.qrun = self.qrun
        other.run = self.run
        other.run_dot = self.run_dot
        other.run_num = self.run_num
        other.pdot = self.pdot
        other.must_call = self.must_call
        other.sig = self.sig
        other.ident = self.ident
        other.ident_dot = self.ident_dot
        other.stack = self.stack.copy()
        other.nargs = self.nargs
        other.progress = self.progress
        other.terminated = self.terminated
        other.violated = self.violated
        return other

    # ------------------------------------------------------------------ rules
    @property
    def path_complete(self) -> bool:
        return self.progress >= len(self.spec.names)

    def _next_name(self) -> str | None:
        names = self.spec.names
        return names[self.progress] if self.progress < len(names) else None

    def _check_attribute(self, name: str) -> bool:
        """A name completed after a dot.  Returns False on violation."""
        spec = self.spec
        lang = spec.lang
        if self.nargs == 0:
            if name == self._next_name():
                if name not in lang.namespace:
                    self.must_call = True
                return True
            if name in lang.namespace:
                return True
            if spec.strict:
                return False
            return name not in lang.callable
        return name in spec.helpers or name in lang.namespace or name not in lang.callable

    def _open_call(self, name: str, after_dot: bool) -> int:
        """Frame kind for ``name(``; -1 on violation."""
        spec = self.spec
        lang = spec.lang
        if after_dot:
            if name not in lang.callable:
                # On the main chain the only attribute names that got this far are
                # namespace names (modules, attributes); calling one is not a path call.
                return -1 if self.nargs == 0 and spec.strict else _CALL
        elif name not in lang.bare_callable:
            return _CALL
        if self.nargs == 0:
            if name == self._next_name():
                self.progress += 1
                return _PATH_LAST if self.progress == len(spec.names) else _PATH
            return -1
        return _CALL if name in spec.helpers else -1

    def _complete_run(self) -> bool:
        name = self.run
        after_dot = self.run_dot
        self.run = None
        if self.run_num:
            self.run_num = False
            self.sig = _SIG_OTHER
            return True
        assert name is not None
        if after_dot and not self._check_attribute(name):
            return False
        if name in KEYWORDS and not after_dot:
            self.sig = _SIG_OTHER
        else:
            self.sig = _SIG_IDENT
            self.ident = name
            self.ident_dot = after_dot
        return True

    # ------------------------------------------------------------------ scanning
    def feed(self, text: str) -> bool:
        """Consume ``text``.  Returns False (and sets ``violated``) on a rule violation.

        Scanning stops as soon as the option terminates; the rest of the text belongs
        to the high-level policy and is not checked.
        """
        if self.violated:
            return False
        for ch in text:
            if self.terminated:
                return True
            lex = self.lex
            if lex == _COMMENT:
                if ch == "\n":
                    self.lex = _NORMAL
                continue
            if lex == _STRING:
                if self.escape:
                    self.escape = False
                elif ch == "\\":
                    self.escape = True
                elif self.triple:
                    if ch == self.quote:
                        self.qrun += 1
                        if self.qrun == 3:
                            self.lex, self.qrun, self.sig = _NORMAL, 0, _SIG_CLOSE
                    else:
                        self.qrun = 0
                elif ch == self.quote or ch == "\n":
                    self.lex, self.sig = _NORMAL, _SIG_CLOSE
                continue
            if lex == _OPENING:
                if ch == self.quote:
                    self.qrun += 1
                    if self.qrun == 3:
                        self.lex, self.triple, self.qrun, self.escape = _STRING, True, 0, False
                    continue
                if self.qrun == 1:  # an ordinary string whose first character is ``ch``
                    self.lex, self.triple, self.qrun = _STRING, False, 0
                    self.escape = ch == "\\"
                    if ch == "\n":
                        self.lex, self.sig = _NORMAL, _SIG_CLOSE
                    continue
                # qrun == 2: the empty string literal just closed; scan ``ch`` normally.
                self.lex, self.qrun, self.sig = _NORMAL, 0, _SIG_CLOSE

            # ---------------------------------------------------------- normal mode
            if _is_ident(ch):
                if self.run is None:
                    if self.must_call:
                        return self._violate()
                    self.run = ch
                    self.run_dot = self.pdot
                    self.run_num = ch.isdigit() and not self.pdot
                    self.pdot = False
                else:
                    self.run += ch
                continue
            if self.run is not None:
                if self.run_num and ch == ".":
                    self.run += ch
                    continue
                if not self._complete_run():
                    return self._violate()
            if ch in _WS:
                if self.pdot:
                    return self._violate()  # "x. name" would escape the attribute check
                continue
            if self.must_call:
                if ch != "(":
                    return self._violate()
                self.must_call = False
            if ch == ".":
                self.pdot = True
                self.sig = _SIG_OTHER
                continue
            self.pdot = False
            if ch == "(":
                if self.sig == _SIG_IDENT:
                    kind = self._open_call(self.ident, self.ident_dot)
                    if kind < 0:
                        return self._violate()
                elif self.sig == _SIG_CLOSE:
                    kind = _CALL
                else:
                    kind = _GROUP
                self.stack.append(kind)
                if kind != _GROUP:
                    self.nargs += 1
                self.sig = _SIG_OTHER
            elif ch == "[" or ch == "{":
                self.stack.append(_GROUP)
                self.sig = _SIG_OTHER
            elif ch == ")" or ch == "]" or ch == "}":
                if self.stack:
                    kind = self.stack.pop()
                    if kind != _GROUP:
                        self.nargs -= 1
                    if kind == _PATH_LAST:
                        self.terminated = True
                self.sig = _SIG_CLOSE
            elif ch == '"' or ch == "'":
                self.lex, self.quote, self.qrun = _OPENING, ch, 1
                self.sig = _SIG_OTHER
            elif ch == "#":
                self.lex = _COMMENT
                self.sig = _SIG_OTHER
            else:
                self.sig = _SIG_OTHER
        return True

    def _violate(self) -> bool:
        self.violated = True
        return False

    def viable(self) -> bool:
        """Can the current state still be completed without a violation?

        A partially typed attribute name on the main chain must be the prefix of a
        name that is allowed there; otherwise every completion would be rejected.
        """
        if self.violated:
            return False
        if self.terminated or self.run is None or not self.run_dot or self.run_num:
            return True
        spec = self.spec
        if self.nargs != 0 or not spec.strict:
            return True
        next_name = self._next_name()
        if next_name is not None and next_name.startswith(self.run):
            return True
        return self.run in spec.lang.namespace_prefixes

    def step(self, text: str) -> bool:
        """Feed ``text`` and report whether the resulting state is admissible."""
        return self.feed(text) and self.viable()

    # ------------------------------------------------------------------ cache key
    def key(self) -> tuple:
        """State reduced to what can influence the admissibility of the next token."""
        lang = self.spec.lang
        run_key: str | None
        if self.run is None:
            run_key = None
        elif self.run_num:
            run_key = "0"
        elif self.run_dot:
            run_key = self.run if self.run in lang.attr_prefixes else "\0"
        else:
            run_key = self.run if self.run in lang.bare_prefixes else "\0"
        ident_key: str | None = None
        if self.sig == _SIG_IDENT:
            known = lang.callable if self.ident_dot else lang.bare_callable
            ident_key = self.ident if self.ident in known else "\0"
        top = tuple(self.stack[-8:])
        deeper = any(kind != _GROUP for kind in self.stack[:-8])
        return (
            self.spec.option_index, self.progress, self.lex, self.quote, self.triple, self.escape,
            self.qrun, run_key, self.run_dot, self.pdot, self.must_call, self.sig, ident_key,
            self.ident_dot, top, deeper,
        )


# --------------------------------------------------------------------------------------
# Vocabulary index
# --------------------------------------------------------------------------------------
class _Lead:
    """Structure-free tokens sharing the same leading identifier fragment."""

    __slots__ = ("rest_empty", "rest_ws", "rest_ws_then", "rest_other")

    def __init__(self) -> None:
        self.rest_empty: list[int] = []  # the token is exactly the fragment
        self.rest_ws: list[int] = []  # fragment followed by whitespace only
        self.rest_ws_then: list[int] = []  # fragment, whitespace, then more text
        self.rest_other: list[int] = []  # fragment followed by a non-space character


class VocabIndex:
    """Token strings of the base vocabulary, indexed for mask compilation."""

    def __init__(self, strings: list[str], control_ids: Iterable[int] = ()) -> None:
        self.strings = strings
        self.size = len(strings)
        control = set(control_ids)
        self.structural: list[int] = []  # tokens that contain a structural character
        self.leads: dict[str, _Lead] = {}
        always_banned: list[int] = []
        for tid, text in enumerate(strings):
            if tid in control or text == "":
                always_banned.append(tid)
                continue
            if any(ch in STRUCTURAL for ch in text):
                self.structural.append(tid)
                continue
            end = 0
            while end < len(text) and _is_ident(text[end]):
                end += 1
            entry = self.leads.get(text[:end])
            if entry is None:
                entry = self.leads[text[:end]] = _Lead()
            rest = text[end:]
            if not rest:
                entry.rest_empty.append(tid)
            elif all(ch in _WS for ch in rest):
                entry.rest_ws.append(tid)
            elif rest[0] in _WS:
                entry.rest_ws_then.append(tid)
            else:
                entry.rest_other.append(tid)
        self.leads.setdefault("", _Lead())
        self.always_banned = always_banned
        self.control = control

    @classmethod
    def from_tokenizer(cls, tokenizer, vocab_size: int | None = None) -> VocabIndex:
        """Decode every token id to the exact text it contributes to the output."""
        size = int(vocab_size if vocab_size is not None else len(tokenizer))
        n_known = min(size, len(tokenizer))
        anchor = tokenizer.encode("a", add_special_tokens=False)[-1:]
        strings: list[str] = [""] * size
        if anchor:
            prefix = tokenizer.decode(anchor, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            batch = 8192
            for start in range(0, n_known, batch):
                ids = [[anchor[0], tid] for tid in range(start, min(start + batch, n_known))]
                decoded = tokenizer.batch_decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                for offset, text in enumerate(decoded):
                    strings[start + offset] = text[len(prefix):] if text.startswith(prefix) else ""
        else:  # pragma: no cover - tokenizers without a plain "a" token
            for tid in range(n_known):
                strings[tid] = tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        control = set(getattr(tokenizer, "all_special_ids", []) or [])
        control.update(getattr(tokenizer, "get_added_vocab", dict)().values())
        control.update(range(n_known, size))  # embedding rows without a token
        return cls(strings, control)


# --------------------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MaskSpec:
    """A token mask over the base vocabulary in sparse form.

    ``mode == "ban"``: every token is allowed except ``ids``.
    ``mode == "allow"``: only ``ids`` are allowed.
    """

    mode: str
    ids: torch.Tensor  # int64, on CPU

    @property
    def empty(self) -> bool:
        return self.mode == "allow" and self.ids.numel() == 0

    def apply_(self, row: torch.Tensor) -> torch.Tensor:
        """Add the mask to a 1-D logits row in place (``-inf`` on forbidden tokens)."""
        ids = self.ids.to(row.device)
        if self.mode == "ban":
            row[ids] = float("-inf")
        else:
            kept = row[ids]
            row.fill_(float("-inf"))
            row[ids] = kept
        return row

    def allows(self, token_id: int) -> bool:
        present = bool((self.ids == token_id).any().item())
        return present if self.mode == "allow" else not present


class ConstraintEngine:
    """Compiles automaton states into token masks and advances automata."""

    def __init__(
        self,
        vocab: VocabIndex,
        lang: LanguageTables,
        options: Iterable[CodeOption],
        strict: bool = True,
        cache_size: int = 50_000,
    ) -> None:
        self.vocab = vocab
        self.lang = lang
        self.strict = strict
        self.specs: dict[int, ConstraintSpec] = {
            option.index: ConstraintSpec.from_option(option, lang, strict) for option in options
        }
        self._cache: OrderedDict[tuple, MaskSpec] = OrderedDict()
        self._cache_size = cache_size
        self._always = set(vocab.always_banned)
        self._structural_text = [(tid, vocab.strings[tid]) for tid in vocab.structural]
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------ lifecycle
    def register(self, option: CodeOption) -> None:
        self.specs[option.index] = ConstraintSpec.from_option(option, self.lang, self.strict)

    def start(self, option_index: int, text_before: str = "") -> OptionAutomaton:
        """Automaton for a freshly initiated option.

        ``text_before`` is the program text preceding the option; if it ends with a dot
        the first name of the option is scanned as an attribute.
        """
        pending_dot = text_before.rstrip(" \t").endswith(".")
        return OptionAutomaton(self.specs[option_index], pending_dot=pending_dot)

    def advance(self, automaton: OptionAutomaton, token_id: int) -> bool:
        """Feed a sampled token.  Returns False if it violates the option (never for masked sampling)."""
        return automaton.step(self.vocab.strings[token_id])

    # ------------------------------------------------------------------ masks
    def mask(self, automaton: OptionAutomaton) -> MaskSpec:
        key = automaton.key()
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.hits += 1
            return cached
        self.misses += 1
        spec = self._compile(automaton)
        self._cache[key] = spec
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return spec

    def _compile(self, state: OptionAutomaton) -> MaskSpec:
        mode, ids = self._structure_free(state)
        simulated_ok: list[int] = []
        simulated_bad: list[int] = []
        for tid, text in self._structural_text:
            if state.copy().step(text):
                simulated_ok.append(tid)
            else:
                simulated_bad.append(tid)
        if mode == "ban":
            banned = set(ids)
            banned.update(simulated_bad)
            banned.update(self._always)
            return MaskSpec("ban", torch.tensor(sorted(banned), dtype=torch.long))
        allowed = set(ids)
        allowed.update(simulated_ok)
        allowed.difference_update(self._always)
        return MaskSpec("allow", torch.tensor(sorted(allowed), dtype=torch.long))

    def _structure_free(self, state: OptionAutomaton) -> tuple[str, list[int]]:
        """Closed-form decision for tokens without structural characters.

        Returns ``("ban", banned_ids)`` or ``("allow", allowed_ids)``.
        """
        leads = self.vocab.leads
        blank = leads[""]
        if state.lex != _NORMAL:
            # Inside a string, a comment or right after an opening quote: plain text
            # cannot violate anything.
            return "ban", []
        if state.run is None:
            if state.must_call:
                return "allow", list(blank.rest_ws)
            if not state.pdot:
                return "ban", []
            run = ""
        else:
            if state.run_num or not state.run_dot:
                return "ban", []
            run = state.run
        spec = state.spec
        lang = spec.lang
        next_name = state._next_name()
        chain_level = state.nargs == 0

        if chain_level and spec.strict:
            allowed: list[int] = []
            candidates: list[tuple[str, bool]] = [(name, False) for name in lang.namespace if name.startswith(run)]
            if next_name is not None and next_name.startswith(run):
                candidates = [c for c in candidates if c[0] != next_name]
                candidates.append((next_name, next_name not in lang.namespace))
            for name, needs_call in candidates:
                remainder = name[len(run):]
                for end in range(1, len(remainder) + 1):
                    entry = leads.get(remainder[:end])
                    if entry is not None:
                        allowed.extend(entry.rest_empty)  # still a prefix of ``name``
                entry = leads.get(remainder)
                if entry is None or (remainder == "" and run == ""):
                    continue
                allowed.extend(entry.rest_ws)
                if not needs_call:
                    allowed.extend(entry.rest_ws_then)
                    allowed.extend(entry.rest_other)
            if run == "":
                allowed.extend(blank.rest_other)  # an operator right after the dot ends the access
            return "allow", allowed

        banned: list[int] = []
        if chain_level:
            forbidden = lang.callable - lang.namespace - ({next_name} if next_name else set())
        else:
            forbidden = lang.callable - lang.namespace - spec.helpers
        for name in forbidden:
            if not name.startswith(run):
                continue
            remainder = name[len(run):]
            if remainder == "" and run == "":
                continue
            entry = leads.get(remainder)
            if entry is not None:  # the token would complete a forbidden name
                banned.extend(entry.rest_ws)
                banned.extend(entry.rest_ws_then)
                banned.extend(entry.rest_other)
        if chain_level and next_name is not None and next_name.startswith(run) and next_name not in lang.namespace:
            entry = leads.get(next_name[len(run):])
            if entry is not None and not (run == "" and next_name == ""):
                banned.extend(entry.rest_ws_then)
                banned.extend(entry.rest_other)
        if run == "":
            banned.extend(blank.rest_ws)  # whitespace right after a dot
            banned.extend(blank.rest_ws_then)
        return "ban", banned

    def cache_info(self) -> dict[str, int]:
        return {"entries": len(self._cache), "hits": self.hits, "misses": self.misses}
