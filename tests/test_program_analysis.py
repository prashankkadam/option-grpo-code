"""Static analysis of generated programs: call sequences, quality metrics and ODL."""

from __future__ import annotations

import math

from option_grpo.analysis.callseq import analyze_program, contiguous_subsequences
from option_grpo.analysis.quality import analyze_quality, analyze_redundancy, detect_reimplementation
from option_grpo.options.odl import cover, decompose, option_description_length
from option_grpo.options.option import CERTIFIED

CONTEXT = 'import tinyframe\ncsv_path = "data.csv"\n'

IDIOMATIC = """df = tinyframe.read_csv(csv_path)
result = df.where(tinyframe.col("salary") > 75000).collect()
"""

SLOP = """import pandas as pd
path = csv_path
df_raw = tinyframe.read_csv(path)
table = df_raw.collect()
result_list = []
for row in table.to_pylist():
    if row["salary"] > 75000:
        result_list.append(row)
result_df = tinyframe.from_records(result_list)
final = result_df.collect()
"""


def _certified_patterns(vocab):
    for option in vocab.options:
        option.status = CERTIFIED
    return vocab.patterns([0, 1])


def test_call_sequence_uses_type_propagation(index):
    analysis = analyze_program(IDIOMATIC, index, CONTEXT)
    assert analysis.ok
    calls = [(c.name, c.role, c.func) for c in analysis.calls]
    assert calls == [
        ("read_csv", "chain", "tinyframe._io.read_csv"),
        ("col", "arg", "tinyframe.expr.col"),
        ("where", "chain", "tinyframe.frame.Frame.where"),
        ("collect", "chain", "tinyframe.frame.Frame.collect"),
    ]
    assert analysis.primary_names() == ["read_csv", "where", "collect"]
    assert analysis.n_api == 4


def test_calls_on_builtin_and_foreign_values_are_not_sdk_calls(index):
    code = "import requests\nitems = []\nitems.append(1)\nr = requests.get('http://x')\ndata = r.json()\nn = len(items)\n"
    analysis = analyze_program(code, index, CONTEXT)
    assert analysis.sdk_calls == []
    assert analysis.foreign_modules == {"requests"}
    assert [c.foreign for c in analysis.calls if c.name in {"get", "json"}] == ["requests", "requests"]


def test_unparseable_program_is_reported(index):
    analysis = analyze_program("result = (", index)
    assert not analysis.ok and analysis.error.startswith("SyntaxError")


def test_contiguous_subsequences():
    assert contiguous_subsequences(["a", "b", "c"], 2, 3) == {("a", "b"), ("b", "c"), ("a", "b", "c")}
    assert contiguous_subsequences(["a"], 2, 4) == set()


def test_redundancy_and_reimplementation_detection(index):
    analysis = analyze_program(SLOP, index, CONTEXT)
    report = analyze_redundancy(analysis.tree)
    assert set(report.redundant) == {"pd", "path"}
    assert report.reasons["pd"] == "unused import" and report.reasons["path"] == "alias"
    assert report.n_vars == 8 and abs(report.q_red - 0.75) < 1e-9
    assert detect_reimplementation(analysis) == ["loop-accumulate"]
    clean = analyze_program(IDIOMATIC, index, CONTEXT)
    assert analyze_redundancy(clean.tree).n_redundant == 0
    assert detect_reimplementation(clean) == []


def test_quality_report_separates_idiomatic_code_from_slop(index, vocab):
    patterns = _certified_patterns(vocab)
    good = analyze_quality(IDIOMATIC, index, patterns, CONTEXT)
    bad = analyze_quality(SLOP, index, patterns, CONTEXT)
    assert good.odl == 1.0 and len(good.options_used) == 1
    assert math.isinf(bad.odl)  # re-implemented by hand: not expressible as options
    assert good.n_tokens < bad.n_tokens
    assert good.idiomaticity > bad.idiomaticity
    assert good.option_reuse == 1.0 and bad.option_reuse < 1.0
    assert good.q_red == 1.0 and bad.q_red < 1.0
    assert good.conciseness(good.n_tokens) == 1.0
    assert good.api_efficiency(4) == 1.0 and bad.api_efficiency(4) < 1.0


def test_odl_is_the_minimum_number_of_options(vocab):
    patterns = _certified_patterns(vocab)
    composed = next(o for o in vocab.level(1) if o.call_names == ("read_csv", "where", "collect"))
    assert option_description_length(["read_csv", "where", "collect"], patterns) == 1.0
    # Without the composed option the same program needs three atomic options.
    atomic_only = {k: v for k, v in patterns.items() if len(k) == 1}
    assert option_description_length(["read_csv", "where", "collect"], atomic_only) == 3.0
    decomposition = decompose(["read_csv", "where", "collect", "to_pylist"], patterns)
    assert decomposition.odl == 2.0 and decomposition.options[0] is composed
    assert math.isinf(option_description_length(["read_csv", "not_an_api"], patterns))
    assert math.isinf(option_description_length(["read_csv"] * 9, atomic_only, max_options=8))
    assert math.isinf(option_description_length(["read_csv"], patterns, expressible=False))
    # ``cover`` tolerates calls that match no option.
    matches = cover(["not_an_api", "read_csv", "where", "collect"], patterns, min_len=2)
    assert [(start, end) for start, end, _ in matches] == [(1, 4)]
