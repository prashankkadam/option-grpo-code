"""Static analysis of generated programs: API-call sequences and code quality."""

from option_grpo.analysis.callseq import CallSite, ProgramAnalysis, analyze_program, api_call_sequences, contiguous_subsequences
from option_grpo.analysis.quality import (
    QualityReport,
    analyze_quality,
    analyze_redundancy,
    detect_reimplementation,
    tokenizer_offsets,
    whitespace_offsets,
)

__all__ = [
    "CallSite",
    "ProgramAnalysis",
    "QualityReport",
    "analyze_program",
    "analyze_quality",
    "analyze_redundancy",
    "api_call_sequences",
    "contiguous_subsequences",
    "detect_reimplementation",
    "tokenizer_offsets",
    "whitespace_offsets",
]
