"""The reward ``R(c) = R_exec(c) + lambda_q R_quality(c) + lambda_c R_consistency(c)``.

* ``R_exec`` is the layered execution reward: 0.0 fails to parse, 0.1 imports fail,
  0.3 runtime error, 0.5 executes, 0.7 output of the expected type, 1.0 passes the
  property-based tests.
* ``R_quality`` bundles the pattern-use, no-redundancy and minimal-API bonuses and the
  re-implementation penalty (``alpha_i = 0.1``).
* ``R_consistency(y_i) = |{y_j : output(y_j) = output(y_i)}| / G`` rewards agreement
  within the sampled group when the property tests are not decisive.

Following Algorithm 4, the quality term is added from Level 1 and the consistency term
from Level 2.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from option_grpo.analysis.callseq import analyze_program
from option_grpo.analysis.quality import OffsetFn, QualityReport, analyze_quality, whitespace_offsets
from option_grpo.config import RewardConfig
from option_grpo.execution.sandbox import ExecResult, Executor
from option_grpo.options.odl import cover
from option_grpo.options.option import CodeOption, OptionVocabulary
from option_grpo.sdk.introspect import SDKIndex
from option_grpo.tasks.dataset import Task


@dataclass
class RewardBreakdown:
    total: float
    exec: float
    quality: float = 0.0
    consistency: float = 0.0
    stage: str = "parse_error"
    passed: bool = False
    flags: dict[str, bool] = field(default_factory=dict)
    report: QualityReport | None = None
    result: ExecResult | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "exec": self.exec,
            "quality": self.quality,
            "consistency": self.consistency,
            "stage": self.stage,
            "passed": self.passed,
            "flags": self.flags,
        }


def execution_reward(stage: str, cfg: RewardConfig) -> float:
    """Layered execution reward for the furthest stage a program reached."""
    if stage in {"parse_error", "empty"}:
        return cfg.r_parse_fail
    if stage in {"import_error", "setup_error"}:
        return cfg.r_import_fail
    if stage in {"runtime_error", "timeout", "crash"}:
        return cfg.r_runtime_error
    if stage in {"output_none", "type_mismatch"}:
        return cfg.r_executes
    if stage == "type_ok":
        return cfg.r_type_match
    if stage == "tests_passed":
        return cfg.r_tests_pass
    raise ValueError(f"unknown execution stage: {stage}")


def quality_reward(
    report: QualityReport,
    n_min: int,
    level1_patterns: dict[tuple[str, ...], CodeOption],
    cfg: RewardConfig,
) -> tuple[float, dict[str, bool]]:
    """``R_quality`` and the indicator of each of its four components."""
    uses_pattern = bool(cover(report.primary, level1_patterns, min_len=2)) if level1_patterns else False
    flags = {
        "pattern": uses_pattern,
        "no_redundancy": report.n_redundant == 0,
        "minimal_api": 0 < report.n_api <= max(n_min, 1),
        "reimplementation": bool(report.reimplementation),
    }
    value = (
        cfg.pattern_bonus * flags["pattern"]
        + cfg.no_redundancy_bonus * flags["no_redundancy"]
        + cfg.minimal_api_bonus * flags["minimal_api"]
        - cfg.reimplementation_penalty * flags["reimplementation"]
    )
    return float(value), flags


def consistency_reward(digests: list[str | None]) -> list[float]:
    """Share of the group that produced the same observable output as each sample."""
    group = len(digests)
    if group == 0:
        return []
    counts: dict[str, int] = {}
    for d in digests:
        if d is not None:
            counts[d] = counts.get(d, 0) + 1
    return [counts[d] / group if d is not None else 0.0 for d in digests]


def precheck(code: str) -> str | None:
    """Stage decided without running anything: ``parse_error``, ``empty`` or None.

    A program without any statement (blank or comments only) is scored like a program
    that does not parse, otherwise emitting nothing would earn the "executes" reward.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return "parse_error"
    return None if tree.body else "empty"


class RewardComputer:
    """Scores groups of sampled programs."""

    def __init__(
        self,
        cfg: RewardConfig,
        executor: Executor,
        index: SDKIndex,
        vocab: OptionVocabulary,
        offsets: OffsetFn = whitespace_offsets,
        max_options: int = 8,
    ) -> None:
        self.cfg = cfg
        self.executor = executor
        self.index = index
        self.vocab = vocab
        self.offsets = offsets
        self.max_options = max_options

    # ------------------------------------------------------------------ patterns
    def level1_patterns(self, level: int) -> dict[tuple[str, ...], CodeOption]:
        """Level-1 patterns that earn the pattern-use bonus at a curriculum level.

        While Level 1 is being learned the inventory is the candidate pool
        ``O_1^cand``; afterwards only certified options count.
        """
        return self.vocab.patterns([1], certified_only=level >= 2)

    def analysis_patterns(self, level: int) -> dict[tuple[str, ...], CodeOption]:
        """All patterns used for static quality analysis at a curriculum level."""
        table = self.vocab.patterns([0], certified_only=level >= 1 and bool(self.vocab.certified(0)))
        table.update(self.level1_patterns(level))
        return table

    # ------------------------------------------------------------------ scoring
    def score(self, groups: list[tuple[Task, list[str]]], level: int) -> list[list[RewardBreakdown]]:
        """Score every program of every group.  ``groups`` holds ``(task, programs)``."""
        cfg = self.cfg
        flat: list[tuple[Task, str]] = []
        stages: list[list[str | None]] = []
        for task, programs in groups:
            row: list[str | None] = []
            for program in programs:
                stage = precheck(program)
                row.append(stage)
                if stage is None:
                    flat.append((task, program))
            stages.append(row)
        results = iter(self.executor.run(flat))
        use_quality = level >= cfg.quality_from_level and cfg.quality_weight > 0
        use_consistency = level >= cfg.consistency_from_level and cfg.consistency_weight > 0
        level1 = self.level1_patterns(level) if use_quality else {}
        patterns = self.analysis_patterns(level) if use_quality else {}
        out: list[list[RewardBreakdown]] = []
        for (task, programs), row in zip(groups, stages):
            breakdowns: list[RewardBreakdown] = []
            digests: list[str | None] = []
            for program, stage in zip(programs, row):
                result: ExecResult | None = None
                if stage is None:
                    result = next(results)
                    stage = result.stage
                r_exec = execution_reward(stage, cfg)
                item = RewardBreakdown(total=r_exec, exec=r_exec, stage=stage, passed=stage == "tests_passed", result=result)
                gated = r_exec >= cfg.quality_gate
                if use_quality and gated:
                    analysis = analyze_program(program, self.index, task.context_code())
                    report = analyze_quality(
                        program, self.index, patterns, task.context_code(), self.offsets, self.max_options, analysis
                    )
                    item.report = report
                    if report.parsed:
                        item.quality, item.flags = quality_reward(report, task.n_min, level1, cfg)
                digests.append(result.output_digest if (result is not None and gated and result.executed) else None)
                breakdowns.append(item)
            if use_consistency:
                for item, value in zip(breakdowns, consistency_reward(digests)):
                    item.consistency = value
            for item in breakdowns:
                item.total = (
                    item.exec
                    + (cfg.quality_weight * item.quality if use_quality else 0.0)
                    + (cfg.consistency_weight * item.consistency if use_consistency else 0.0)
                )
            out.append(breakdowns)
        return out
