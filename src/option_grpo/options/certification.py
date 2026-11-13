"""Option certification from a trained policy (Algorithm 2).

After a curriculum phase, solutions are sampled from the trained policy, the successful
ones are parsed into API-call sequences and the contiguous sub-sequences that recur
across tasks are identified.  A candidate option is *certified* when its call path is
among those recurring patterns: the policy has demonstrated that it can invoke the
candidate correctly.  Only certified options enter the next phase's action space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from option_grpo.analysis.callseq import contiguous_subsequences
from option_grpo.config import CertificationConfig
from option_grpo.graphs.build import SDKGraphs
from option_grpo.options.discovery import cluster_paths, helper_names
from option_grpo.options.option import CANDIDATE, CERTIFIED, REJECTED, CodeOption, OptionVocabulary, make_option_key, option_io_types
from option_grpo.utils import get_logger

log = get_logger(__name__)


@dataclass
class SolutionRecord:
    """One sampled solution: its task, execution reward and chain-level API calls."""

    task_id: str
    reward: float  # R_exec
    calls: list[str]  # call names, in evaluation order
    funcs: list[str | None] = field(default_factory=list)  # resolved function ids, aligned with ``calls``


@dataclass
class PatternStats:
    successes: int = 0  # successful solutions containing the pattern
    occurrences: int = 0  # all solutions containing the pattern
    tasks: set[str] = field(default_factory=set)  # distinct tasks with a successful occurrence

    @property
    def success_rate(self) -> float:
        return self.successes / self.occurrences if self.occurrences else 0.0


def mine_patterns(
    records: list[SolutionRecord],
    min_len: int,
    max_len: int,
    success_threshold: float,
) -> dict[tuple[str, ...], PatternStats]:
    """Frequency statistics of contiguous call sub-sequences (Algorithm 2, lines 2-5)."""
    stats: dict[tuple[str, ...], PatternStats] = {}
    for record in records:
        success = record.reward >= success_threshold
        for pattern in contiguous_subsequences(record.calls, min_len, max_len):
            entry = stats.setdefault(pattern, PatternStats())
            entry.occurrences += 1
            if success:
                entry.successes += 1
                entry.tasks.add(record.task_id)
    return stats


def _function_successes(
    records: list[SolutionRecord], success_threshold: float
) -> tuple[dict[str, PatternStats], dict[str, PatternStats]]:
    """Per-call statistics for Level 0.

    Returns statistics keyed by resolved function id and, for calls the static analysis
    could not attribute to a single function, statistics keyed by call name.
    """
    by_id: dict[str, PatternStats] = {}
    by_name: dict[str, PatternStats] = {}
    for record in records:
        success = record.reward >= success_threshold
        funcs = record.funcs if len(record.funcs) == len(record.calls) else [None] * len(record.calls)
        seen: set[tuple[str, str]] = set()
        for name, func in zip(record.calls, funcs):
            key = ("id", func) if func else ("name", name)
            if key in seen:
                continue
            seen.add(key)
            table = by_id if func else by_name
            entry = table.setdefault(key[1], PatternStats())
            entry.occurrences += 1
            if success:
                entry.successes += 1
                entry.tasks.add(record.task_id)
    return by_id, by_name


def certify_options(
    vocab: OptionVocabulary,
    level: int,
    records: list[SolutionRecord],
    cfg: CertificationConfig | None = None,
    graphs: SDKGraphs | None = None,
) -> dict[str, Any]:
    """Certify the candidates of ``level`` in place and return summary statistics.

    Level 0 uses single calls, Level 1 contiguous sub-sequences of 2-4 calls.  A pattern
    qualifies when it occurs in at least ``nu`` successful solutions spread over enough
    distinct tasks (one task per function exists at Level 0, so the task requirement is
    ``level0_min_tasks`` there and ``nu`` at Level 1).
    """
    cfg = cfg or CertificationConfig()
    min_len, max_len = (1, 1) if level == 0 else (2, 4)
    min_tasks = cfg.level0_min_tasks if level == 0 else cfg.frequency_threshold
    stats = mine_patterns(records, min_len, max_len, cfg.success_threshold)

    def qualifies(entry: PatternStats | None) -> bool:
        return (
            entry is not None
            and entry.successes >= cfg.frequency_threshold
            and len(entry.tasks) >= min_tasks
            and entry.success_rate >= cfg.min_invocation_success_rate
        )

    by_function, by_name = _function_successes(records, cfg.success_threshold) if level == 0 else ({}, {})
    index = graphs.index if graphs is not None else None
    certified = 0
    matched_patterns: set[tuple[str, ...]] = set()
    for option in vocab.level(level):
        if option.status not in {CANDIDATE, CERTIFIED, REJECTED}:
            continue
        entry: PatternStats | None = None
        if level == 0:
            # Statistics of the exact function; calls that could not be attributed to a
            # single function are matched by call name.
            entry = by_function.get(option.path[0]) or by_name.get(option.call_names[0])
        else:
            names = {option.call_names}
            if index is not None:
                names.update(tuple(index.functions[f].name for f in member if f in index.functions) for member in option.members)
            best: PatternStats | None = None
            for pattern in names:
                candidate = stats.get(pattern)
                if candidate is not None and (best is None or candidate.successes > best.successes):
                    best = candidate
                    if qualifies(candidate):
                        matched_patterns.add(pattern)
            entry = best
        if qualifies(entry):
            option.status = CERTIFIED
            certified += 1
        else:
            option.status = REJECTED
        if entry is not None:
            option.stats.update(
                successes=entry.successes,
                occurrences=entry.occurrences,
                tasks=len(entry.tasks),
                success_rate=round(entry.success_rate, 4),
            )

    novel = 0
    if cfg.allow_novel_patterns and level >= 1 and graphs is not None:
        novel = _add_novel_patterns(vocab, graphs, stats, matched_patterns, qualifies)
    if graphs is not None:
        vocab.build_composability(graphs.index)
    summary = {
        "level": level,
        "solutions": len(records),
        "successful": sum(1 for r in records if r.reward >= cfg.success_threshold),
        "candidates": len(vocab.level(level)),
        "certified": certified + novel,
        "novel": novel,
        "patterns_mined": len(stats),
    }
    log.info("certification: %s", summary)
    return summary


def _add_novel_patterns(vocab, graphs, stats, matched, qualifies) -> int:
    """Turn recurring patterns that match no candidate into new Level-1 options.

    The patterns are clustered by semantic similarity (Algorithm 2, line 7) and the most
    frequent pattern of every cluster becomes an option.
    """
    index = graphs.index
    known = {o.call_names for o in vocab.level(1)}
    fresh: list[tuple[tuple[str, ...], list[Any], Any]] = []
    for pattern, entry in stats.items():
        if pattern in matched or pattern in known or not qualifies(entry):
            continue
        functions = []
        for name in pattern:
            candidates = index.by_name(name)
            if not candidates:
                break
            functions.append(candidates[0])
        else:
            if all(graphs.flow.has_edge(a.id, b.id) for a, b in zip(functions, functions[1:])):
                fresh.append((pattern, functions, entry))
    if not fresh:
        return 0
    embeddings = np.stack([np.mean([graphs.embeddings[f.id] for f in functions], axis=0) for _, functions, _ in fresh])
    labels = cluster_paths(embeddings, 0.8)
    added = 0
    for label in sorted(set(labels.tolist())):
        members = [fresh[i] for i in range(len(fresh)) if labels[i] == label]
        pattern, functions, entry = max(members, key=lambda item: item[2].successes)
        key = make_option_key(1, functions)
        if vocab.get(key) is not None:
            continue
        in_types, out_types, is_source = option_io_types(functions[0], functions[-1])
        vocab.add(
            CodeOption(
                index=-1,
                level=1,
                key=key,
                path=tuple(f.id for f in functions),
                call_names=pattern,
                in_types=in_types,
                out_types=out_types,
                is_source=is_source,
                helpers=helper_names(functions, graphs),
                weight=1.0,
                members=[tuple(f.id for f in fs) for _, fs, _ in members],
                status=CERTIFIED,
                stats={"novel": True, "successes": entry.successes, "tasks": len(entry.tasks)},
            )
        )
        added += 1
    return added
