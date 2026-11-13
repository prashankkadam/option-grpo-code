"""Evaluation: pass@k, code-quality metrics, baselines and analyses."""

from option_grpo.eval.evaluator import EvalReport, Evaluator, PolicyGenerator, SampleRecord, odl_quality_correlations
from option_grpo.eval.stats import pass_at_k, spearman

__all__ = [
    "EvalReport",
    "Evaluator",
    "PolicyGenerator",
    "SampleRecord",
    "odl_quality_correlations",
    "pass_at_k",
    "spearman",
]
