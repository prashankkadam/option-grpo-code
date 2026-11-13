"""Evaluation on UnseenSDK-Bench.

For every task ``n`` programs are sampled (``n = 10`` at temperature 0.8 by default) and
executed.  Reported metrics:

* ``pass@k``     -- functional correctness (unbiased estimator over the ``n`` samples);
* ``CL``         -- code length: mean token count of passing solutions (lower is better);
* ``AE``         -- API efficiency: mean number of SDK calls in passing solutions (lower);
* ``ID``         -- idiomaticity: share of tokens covered by discovered options (higher);
* ``OR``         -- option reuse: share of SDK calls inside a matched Level-1 option;
* ``Q_conc``, ``Q_api``, ``Q_red`` and the option description length of Definition 4.

The same evaluator scores the hierarchical policy and every baseline: a *generator*
turns a task into ``n`` candidate programs, everything downstream is shared.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from option_grpo.analysis.callseq import analyze_program
from option_grpo.analysis.quality import OffsetFn, analyze_quality, whitespace_offsets
from option_grpo.config import EvalConfig, RewardConfig
from option_grpo.eval.stats import pass_at_k, spearman
from option_grpo.execution.sandbox import Executor
from option_grpo.options.option import CodeOption, OptionVocabulary
from option_grpo.rewards.reward import execution_reward, precheck
from option_grpo.sdk.introspect import SDKIndex
from option_grpo.tasks.dataset import Task
from option_grpo.tasks.prompts import build_prompt, extract_solution
from option_grpo.utils import get_logger, write_json, write_jsonl

log = get_logger(__name__)


class Generator(Protocol):
    """Anything that can propose ``n`` programs for a task."""

    def generate(self, task: Task, n: int, temperature: float, max_new_tokens: int) -> list[str]: ...


class PolicyGenerator:
    """Programs sampled from a :class:`~option_grpo.policy.HierarchicalPolicy`.

    With ``initiation=None`` decoding is token level, which is how the frozen model and
    the flat baselines are evaluated; with an initiation table the options of the
    corresponding action space are available.
    """

    def __init__(self, engine, initiation=None, context_fn: Callable[[Task], str] | None = None, chunk: int = 16) -> None:
        self.engine = engine
        self.initiation = initiation
        self.context_fn = context_fn
        self.chunk = max(chunk, 1)
        self.last_trajectories: list[Any] = []

    def generate(self, task: Task, n: int, temperature: float, max_new_tokens: int) -> list[str]:
        prompt = build_prompt(task, self.context_fn(task) if self.context_fn else "")
        trajectories: list[Any] = []
        while len(trajectories) < n:
            trajectories += self.engine.rollout(
                task.id, prompt, min(self.chunk, n - len(trajectories)), self.initiation,
                task.setup_value_types(), temperature, max_new_tokens,
            )
        self.last_trajectories = trajectories
        return [extract_solution(t.text) for t in trajectories]


@dataclass
class SampleRecord:
    task_id: str
    level: int
    program: str
    stage: str
    passed: bool
    reward_exec: float
    n_tokens: int = 0
    n_api: int = 0
    odl: float | None = None
    idiomaticity: float = 0.0
    option_reuse: float = 0.0
    q_conc: float | None = None
    q_api: float | None = None
    q_red: float | None = None
    n_redundant: int = 0
    reimplementation: list[str] = field(default_factory=list)
    option_decisions: int = 0

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class EvalReport:
    sdk: str
    n_tasks: int
    n_samples: int
    metrics: dict[str, float]
    by_level: dict[str, dict[str, float]]
    records: list[SampleRecord]

    def to_json(self) -> dict[str, Any]:
        def clean(metrics: dict[str, float]) -> dict[str, float | None]:
            # Metrics over passing solutions are undefined when nothing passes.
            return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in metrics.items()}

        return {
            "sdk": self.sdk,
            "n_tasks": self.n_tasks,
            "n_samples": self.n_samples,
            "metrics": clean(self.metrics),
            "by_level": {level: clean(metrics) for level, metrics in self.by_level.items()},
        }

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        write_json(directory / "report.json", self.to_json())
        write_jsonl(directory / "samples.jsonl", (r.to_json() for r in self.records))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


class Evaluator:
    def __init__(
        self,
        index: SDKIndex,
        vocab: OptionVocabulary,
        executor: Executor,
        cfg: EvalConfig | None = None,
        offsets: OffsetFn = whitespace_offsets,
        max_options: int = 8,
        patterns: dict[tuple[str, ...], CodeOption] | None = None,
    ) -> None:
        self.index = index
        self.vocab = vocab
        self.executor = executor
        self.cfg = cfg or EvalConfig()
        self.offsets = offsets
        self.max_options = max_options
        # Quality is measured against the certified options when a trained vocabulary is
        # available and against the candidate inventory otherwise (baselines evaluated
        # before any Option-GRPO run exists).
        if patterns is None:
            certified = bool(vocab.certified())
            patterns = vocab.patterns([0, 1], certified_only=certified)
        self.patterns = patterns
        self._reward_cfg = RewardConfig()

    # ------------------------------------------------------------------ scoring
    def score_task(self, task: Task, programs: list[str], decisions: list[int] | None = None) -> list[SampleRecord]:
        stages = [precheck(p) for p in programs]
        runnable = [(task, p) for p, s in zip(programs, stages) if s is None]
        results = iter(self.executor.run(runnable))
        t_ref = len(self.offsets(task.reference)) if task.reference else None
        records: list[SampleRecord] = []
        for position, (program, stage) in enumerate(zip(programs, stages)):
            if stage is None:
                stage = next(results).stage
            record = SampleRecord(
                task_id=task.id,
                level=task.level,
                program=program,
                stage=stage,
                passed=stage == "tests_passed",
                reward_exec=execution_reward(stage, self._reward_cfg),
                option_decisions=decisions[position] if decisions else 0,
            )
            if stage not in {"parse_error", "empty"}:
                analysis = analyze_program(program, self.index, task.context_code())
                report = analyze_quality(
                    program, self.index, self.patterns, task.context_code(), self.offsets, self.max_options, analysis
                )
                record.n_tokens = report.n_tokens
                record.n_api = report.n_api
                record.odl = None if math.isinf(report.odl) else report.odl
                record.idiomaticity = report.idiomaticity
                record.option_reuse = report.option_reuse
                record.q_conc = report.conciseness(t_ref)
                record.q_api = report.api_efficiency(task.n_min)
                record.q_red = report.q_red
                record.n_redundant = report.n_redundant
                record.reimplementation = report.reimplementation
            records.append(record)
        return records

    # ------------------------------------------------------------------ aggregation
    def aggregate(self, per_task: dict[str, list[SampleRecord]]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if not per_task:
            return metrics
        for k in self.cfg.ks:
            values = []
            for records in per_task.values():
                if len(records) >= k:
                    values.append(pass_at_k(len(records), sum(r.passed for r in records), k))
            if values:
                metrics[f"pass@{k}"] = 100.0 * _mean(values)
        passing = [r for records in per_task.values() for r in records if r.passed]
        everything = [r for records in per_task.values() for r in records]
        metrics["CL"] = _mean([float(r.n_tokens) for r in passing])
        metrics["AE"] = _mean([float(r.n_api) for r in passing])
        metrics["ID"] = _mean([r.idiomaticity for r in passing])
        metrics["OR"] = _mean([r.option_reuse for r in passing])
        metrics["Q_conc"] = _mean([r.q_conc for r in passing if r.q_conc is not None])
        metrics["Q_api"] = _mean([r.q_api for r in passing if r.q_api is not None])
        metrics["Q_red"] = _mean([r.q_red for r in passing if r.q_red is not None])
        finite = [r.odl for r in passing if r.odl is not None]
        metrics["ODL"] = _mean([float(v) for v in finite])
        metrics["ODL_finite_share"] = len(finite) / len(passing) if passing else math.nan
        metrics["reward_exec"] = _mean([r.reward_exec for r in everything])
        metrics["passing_solutions"] = float(len(passing))
        return metrics

    def evaluate(self, tasks: list[Task], generator: Generator, sdk: str | None = None) -> EvalReport:
        per_task: dict[str, list[SampleRecord]] = {}
        for number, task in enumerate(tasks, start=1):
            programs = generator.generate(task, self.cfg.n_samples, self.cfg.temperature, self.cfg.max_new_tokens)
            decisions = None
            trajectories = getattr(generator, "last_trajectories", None)
            if trajectories and len(trajectories) == len(programs):
                decisions = [len(t.calls) for t in trajectories]
            per_task[task.id] = self.score_task(task, programs, decisions)
            if number % 20 == 0 or number == len(tasks):
                done = [r for records in per_task.values() for r in records]
                log.info("evaluated %d/%d tasks | pass rate %.3f", number, len(tasks), _mean([float(r.passed) for r in done]))
        by_level: dict[str, dict[str, float]] = {}
        for level in sorted({t.level for t in tasks}):
            ids = {t.id for t in tasks if t.level == level}
            by_level[f"L{level}"] = self.aggregate({k: v for k, v in per_task.items() if k in ids})
        return EvalReport(
            sdk=sdk or self.index.name,
            n_tasks=len(tasks),
            n_samples=self.cfg.n_samples,
            metrics=self.aggregate(per_task),
            by_level=by_level,
            records=[r for records in per_task.values() for r in records],
        )


def odl_quality_correlations(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Spearman correlation between ODL and each quality metric over passing solutions.

    This is the empirical check of the compression results: shorter option
    descriptions should go with more concise, more API-efficient, less redundant and
    more idiomatic programs (negative correlations).
    """
    passing = [r for r in records if r.get("passed") and r.get("odl") is not None]
    out: dict[str, dict[str, float]] = {}
    for key, label in (("q_conc", "conciseness"), ("q_api", "api_efficiency"), ("q_red", "redundancy"), ("idiomaticity", "idiomaticity")):
        pairs = [(float(r["odl"]), float(r[key])) for r in passing if r.get(key) is not None]
        if len(pairs) < 3:
            out[label] = {"rho": math.nan, "p": math.nan, "n": float(len(pairs))}
            continue
        rho, p = spearman([a for a, _ in pairs], [b for _, b in pairs])
        out[label] = {"rho": rho, "p": p, "n": float(len(pairs))}
    return out
