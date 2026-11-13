"""Code-quality analysis: redundancy, re-implementation and the metrics of Definition 4.

* conciseness   ``Q_conc  = T_ref / |c|``
* API efficiency ``Q_api  = N_min / N_api``
* redundancy    ``Q_red   = 1 - N_redundant / N_vars``
* idiomaticity  ``Q_idiom = |tokens covered by options| / |c|``
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from option_grpo.analysis.callseq import CallSite, ProgramAnalysis, analyze_program
from option_grpo.options.odl import cover, decompose
from option_grpo.options.option import CodeOption
from option_grpo.sdk.introspect import SDKIndex

_CONVERSIONS = {"list", "dict", "tuple", "set", "array", "asarray", "DataFrame", "Series", "tolist", "to_list"}

# Tokeniser interface used for |c| and for idiomaticity: text -> list of (start, end) offsets.
OffsetFn = Callable[[str], "list[tuple[int, int]]"]


def whitespace_offsets(text: str) -> list[tuple[int, int]]:
    """Fallback tokenisation: maximal runs of identifier characters and single symbols."""
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        j = i + 1
        if ch == "_" or ch.isalnum():
            while j < n and (text[j] == "_" or text[j].isalnum()):
                j += 1
        spans.append((i, j))
        i = j
    return spans


def tokenizer_offsets(tokenizer) -> OffsetFn:
    """Offset function backed by a (fast) Hugging Face tokenizer."""

    def offsets(text: str) -> list[tuple[int, int]]:
        if not text:
            return []
        try:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            return [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
        except (NotImplementedError, KeyError, TypeError):
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            return [(0, 0)] * len(ids)

    return offsets


# --------------------------------------------------------------------------------------
# Redundancy
# --------------------------------------------------------------------------------------
@dataclass
class RedundancyReport:
    n_vars: int = 0
    redundant: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def n_redundant(self) -> int:
        return len(self.redundant)

    @property
    def q_red(self) -> float:
        return 1.0 - self.n_redundant / self.n_vars if self.n_vars else 1.0


class _Usage(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stores: dict[str, list[ast.AST]] = {}
        self.loads: dict[str, int] = {}
        self.imports: dict[str, ast.AST] = {}
        self.values: dict[str, ast.expr] = {}

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loads[node.id] = self.loads.get(node.id, 0) + 1
        elif isinstance(node.ctx, ast.Store):
            self.stores.setdefault(node.id, []).append(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)
            if isinstance(target, ast.Name):
                self.values[target.id] = node.value

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports[alias.asname or alias.name.split(".")[0]] = node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.imports[alias.asname or alias.name] = node


def analyze_redundancy(tree: ast.Module, output_names: set[str] | None = None) -> RedundancyReport:
    """Count variables and flag the redundant ones.

    A binding is redundant when it is (i) never read (dead store or unused import),
    (ii) a plain alias of another name, or (iii) a single-use intermediate produced by
    a conversion call (``list(...)``, ``pd.DataFrame(...)``, ``np.array(...)``, ...).
    The variable holding the program's output is never counted as redundant.
    """
    usage = _Usage()
    usage.visit(tree)
    output_names = set(output_names or ())
    last = output_variable(tree)
    if last is not None:
        output_names.add(last)
    report = RedundancyReport()
    names = list(usage.stores) + [n for n in usage.imports if n not in usage.stores]
    report.n_vars = len(names)
    for name in names:
        if name in output_names or name == "_":
            continue
        loads = usage.loads.get(name, 0)
        if name in usage.imports and name not in usage.stores:
            if loads == 0:
                report.redundant.append(name)
                report.reasons[name] = "unused import"
            continue
        if loads == 0:
            report.redundant.append(name)
            report.reasons[name] = "never read"
            continue
        value = usage.values.get(name)
        if isinstance(value, ast.Name):
            report.redundant.append(name)
            report.reasons[name] = "alias"
            continue
        if loads == 1 and isinstance(value, ast.Call):
            callee = value.func
            callee_name = callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", "")
            if callee_name in _CONVERSIONS:
                report.redundant.append(name)
                report.reasons[name] = "single-use conversion"
    return report


def output_variable(tree: ast.Module) -> str | None:
    """Name bound by the last top-level statement, if it is a simple assignment."""
    for stmt in reversed(tree.body):
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Pass)):
            continue
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
            return stmt.targets[0].id
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            return stmt.target.id
        return None
    return None


# --------------------------------------------------------------------------------------
# Re-implementation
# --------------------------------------------------------------------------------------
def detect_reimplementation(analysis: ProgramAnalysis) -> list[str]:
    """Signs that SDK functionality was re-implemented by hand.

    * ``loop-accumulate`` -- an explicit loop that builds a collection element by element;
    * ``foreign-library`` -- calls into a third-party library other than the SDK;
    * ``local-definition`` -- helper functions or classes defined inside the solution;
    * ``dict-mutation`` -- configuration objects assembled by repeated item assignment.
    """
    if analysis.tree is None:
        return []
    found: list[str] = []
    if analysis.foreign_modules:
        found.append("foreign-library")
    item_assignments: dict[str, int] = {}
    for node in ast.walk(analysis.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if "local-definition" not in found:
                found.append("local-definition")
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            if _accumulates(node) and "loop-accumulate" not in found:
                found.append("loop-accumulate")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    item_assignments[target.value.id] = item_assignments.get(target.value.id, 0) + 1
    if any(count >= 2 for count in item_assignments.values()):
        found.append("dict-mutation")
    return found


def _accumulates(loop: ast.For | ast.AsyncFor | ast.While) -> bool:
    for node in ast.walk(loop):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"append", "extend", "add", "update", "insert"}:
                return True
        if isinstance(node, ast.AugAssign):
            return True
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Subscript) for t in node.targets):
            return True
    return False


# --------------------------------------------------------------------------------------
# Option coverage and the combined report
# --------------------------------------------------------------------------------------
def _char_offset(lines: list[int], lineno: int, col: int) -> int:
    return lines[min(lineno - 1, len(lines) - 1)] + col


def covered_spans(code: str, calls: list[CallSite]) -> list[tuple[int, int]]:
    """Merged character spans of the given call expressions."""
    starts = [0]
    for line in code.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    encoded_lines = code.splitlines(keepends=True)

    def to_char(lineno: int, col: int) -> int:
        # ``col`` is a UTF-8 byte offset; convert it to a character offset.
        line = encoded_lines[lineno - 1] if 0 < lineno <= len(encoded_lines) else ""
        prefix = line.encode("utf-8")[:col].decode("utf-8", errors="ignore")
        return starts[min(lineno - 1, len(starts) - 1)] + len(prefix)

    spans = sorted((to_char(c.lineno, c.col), to_char(c.end_lineno, c.end_col)) for c in calls)
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


@dataclass
class QualityReport:
    """Static quality profile of one program."""

    parsed: bool
    n_tokens: int = 0  # |c|
    n_api: int = 0  # N_api
    n_vars: int = 0
    n_redundant: int = 0
    redundant: list[str] = field(default_factory=list)
    reimplementation: list[str] = field(default_factory=list)
    foreign_modules: list[str] = field(default_factory=list)
    primary: list[str] = field(default_factory=list)  # chain-level SDK calls
    odl: float = math.inf
    options_used: list[str] = field(default_factory=list)  # keys of the options in the ODL decomposition
    pattern_matches: list[str] = field(default_factory=list)  # keys of matched Level-1 options
    idiomaticity: float = 0.0  # Q_idiom
    option_reuse: float = 0.0  # share of SDK calls inside a matched Level-1 option
    q_red: float = 1.0

    def conciseness(self, t_ref: int | None) -> float | None:
        if not t_ref or not self.n_tokens:
            return None
        return t_ref / self.n_tokens

    def api_efficiency(self, n_min: int | None) -> float | None:
        if not n_min or not self.n_api:
            return None
        return n_min / self.n_api

    def to_json(self, t_ref: int | None = None, n_min: int | None = None) -> dict[str, Any]:
        return {
            "n_tokens": self.n_tokens,
            "n_api": self.n_api,
            "n_vars": self.n_vars,
            "n_redundant": self.n_redundant,
            "reimplementation": self.reimplementation,
            "primary": self.primary,
            "odl": None if math.isinf(self.odl) else self.odl,
            "options_used": self.options_used,
            "idiomaticity": self.idiomaticity,
            "option_reuse": self.option_reuse,
            "q_red": self.q_red,
            "q_conc": self.conciseness(t_ref),
            "q_api": self.api_efficiency(n_min),
        }


def analyze_quality(
    code: str,
    index: SDKIndex,
    patterns: dict[tuple[str, ...], CodeOption],
    context: str = "",
    offsets: OffsetFn = whitespace_offsets,
    max_options: int = 8,
    analysis: ProgramAnalysis | None = None,
) -> QualityReport:
    """Compute every static quality quantity of a program.

    ``patterns`` maps call-name sequences to the options they realise (usually the
    certified options of Levels 0 and 1).
    """
    analysis = analysis or analyze_program(code, index, context)
    token_spans = offsets(code)
    if not analysis.ok or analysis.tree is None:
        return QualityReport(parsed=False, n_tokens=len(token_spans))
    redundancy = analyze_redundancy(analysis.tree)
    reimplementation = detect_reimplementation(analysis)
    primary = analysis.primary
    names = [c.name for c in primary]
    expressible = not reimplementation
    decomposition = decompose(names, patterns, max_options) if expressible and names else None
    odl = decomposition.odl if decomposition is not None else math.inf

    matches = cover(names, patterns, min_len=1)
    covered_calls = [primary[i] for start, end, _ in matches for i in range(start, end)]
    spans = covered_spans(code, covered_calls)
    covered_tokens = 0
    if spans and token_spans and any(b > a for a, b in token_spans):
        for start, end in token_spans:
            if any(start < span_end and end > span_start for span_start, span_end in spans):
                covered_tokens += 1
        idiomaticity = covered_tokens / len(token_spans)
    elif spans:
        total = sum(1 for ch in code if not ch.isspace())
        inside = sum(sum(1 for ch in code[a:b] if not ch.isspace()) for a, b in spans)
        idiomaticity = inside / total if total else 0.0
    else:
        idiomaticity = 0.0

    level1 = {names_: opt for names_, opt in patterns.items() if opt.level >= 1}
    reuse_matches = cover(names, level1, min_len=2)
    reused = sum(end - start for start, end, _ in reuse_matches)
    return QualityReport(
        parsed=True,
        n_tokens=len(token_spans),
        n_api=analysis.n_api,
        n_vars=redundancy.n_vars,
        n_redundant=redundancy.n_redundant,
        redundant=redundancy.redundant,
        reimplementation=reimplementation,
        foreign_modules=sorted(analysis.foreign_modules),
        primary=names,
        odl=odl,
        options_used=[opt.key for opt in decomposition.options] if decomposition and decomposition.finite else [],
        pattern_matches=[opt.key for _, _, opt in reuse_matches],
        idiomaticity=idiomaticity,
        option_reuse=reused / len(names) if names else 0.0,
        q_red=redundancy.q_red,
    )
