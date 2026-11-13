"""Task generation, property tests, sandboxed execution and the reward."""

from __future__ import annotations

import pytest

from option_grpo.config import RewardConfig
from option_grpo.options.option import CERTIFIED
from option_grpo.rewards.reward import RewardComputer, consistency_reward, execution_reward, precheck
from option_grpo.tasks.prompts import BEGIN_MARKER, build_prompt, extract_solution


def _task(tasks, *names):
    return next(t for t in tasks if t.call_names == list(names))


def test_curriculum_levels_and_splits(tasks):
    counts = tasks.counts()
    assert {key.split("/")[0] for key in counts} == {"L0", "L1", "L2"}
    assert all(t.length == 1 for t in tasks.select(level=0))
    assert all(2 <= t.length <= 4 for t in tasks.select(level=1))
    assert all(t.length >= 3 for t in tasks.select(level=2))
    assert tasks.select(split="train") and tasks.select(split="test")
    assert len({t.id for t in tasks}) == len(tasks)
    assert all(t.validated and t.reference and t.n_min >= t.length for t in tasks)


def test_prompt_format_and_solution_extraction(tasks):
    task = _task(tasks, "read_csv")
    prompt = build_prompt(task)
    assert prompt.startswith('"""Problem: Read a CSV file into a lazy frame.')
    assert "import tinyframe" in prompt and "# Setup" in prompt and prompt.endswith(BEGIN_MARKER + "\n")
    assert task.reference.strip() not in prompt  # the skeleton is never shown
    assert extract_solution("result = 1\n# END SOLUTION\nignored") == "result = 1\n"
    assert extract_solution("```python\nx = 2\n```\n") == "x = 2\n"
    assert extract_solution("   \n") == ""
    with_context = build_prompt(task, "# retrieved\n\n")
    assert with_context.startswith("# retrieved")


def test_property_tests_follow_algorithm_3(tasks):
    task = _task(tasks, "read_csv")
    kinds = [t["kind"] for t in task.tests]
    assert kinds.count("type") == 1 and "determinism" in kinds and "call_type" in kinds
    assert {v["name"] for v in task.variants} == {"empty", "single"}
    assert all(t.get("expect") == "ok" for t in task.tests if t["kind"] == "edge")
    # Function-level properties exercised by the skeleton are required.
    assert all(t.get("required") for t in task.tests if t["kind"] == "call_type")
    limit = next(t for t in tasks.select(level=0) if t.call_names == ["limit"] and "Frame" in t.path[0])
    assert any(t["kind"] == "invariant" and t["expr"] == "n >= 0" for t in limit.tests)


@pytest.mark.parametrize(
    "program, stage",
    [
        ("def (:\n", "parse_error"),
        ("import module_that_does_not_exist\nresult = 1\n", "import_error"),
        ("result = 1 / 0\n", "runtime_error"),
        ("frame = tinyframe.read_csv(path)\nprint(frame)\n", "output_none"),
        ("result = [1, 2, 3]\n", "type_mismatch"),
        ("while True:\n    pass\n", "timeout"),
    ],
)
def test_execution_stages(tasks, executor, program, stage, monkeypatch):
    task = _task(tasks, "read_csv")
    if stage == "timeout":
        monkeypatch.setattr(executor.cfg, "timeout_s", 2.0)
    (result,) = executor.run([(task, program)])
    assert result.stage == stage
    assert result.error is None or result.error["type"]


def test_reference_passes_and_calls_are_traced(tasks, executor):
    task = next(t for t in tasks.select(level=1) if t.call_names[0] == "read_csv")
    (result,) = executor.run([(task, task.reference)])
    assert result.stage == "tests_passed" and result.passed
    traced = [c["id"] for c in result.trace if c["id"]]
    assert traced[: len(task.path)] == task.path or set(task.path) <= set(traced)
    assert result.n_api >= task.length
    assert result.output["digest"] and all(t["passed"] for t in result.tests)


def test_returning_an_input_does_not_pass(tasks, executor):
    """A program that never calls the function under test cannot satisfy its properties."""
    task = next(t for t in tasks.select(level=0) if t.call_names == ["where"])
    (result,) = executor.run([(task, "result = frame\n")])
    assert result.stage == "type_ok"
    assert any(t["detail"] == "the function under test was not called" for t in result.tests)


def test_network_is_disabled_in_the_sandbox(tasks, executor):
    task = _task(tasks, "read_csv")
    program = "import socket\ns = socket.socket()\ns.connect(('127.0.0.1', 9))\nresult = 1\n"
    (result,) = executor.run([(task, program)])
    assert result.stage == "runtime_error" and result.error["type"] == "PermissionError"


def test_layered_reward_values():
    cfg = RewardConfig()
    stages = ["parse_error", "import_error", "runtime_error", "output_none", "type_ok", "tests_passed"]
    assert [execution_reward(s, cfg) for s in stages] == [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    assert execution_reward("type_mismatch", cfg) == 0.5 and execution_reward("timeout", cfg) == 0.3
    assert precheck("x = (") == "parse_error" and precheck("# only a comment\n") == "empty" and precheck("x = 1\n") is None
    assert consistency_reward(["a", "a", "b", None]) == [0.5, 0.5, 0.25, 0.0]


def test_reward_terms_are_switched_on_by_level(tasks, executor, index, vocab):
    for option in vocab.options:
        option.status = CERTIFIED
    computer = RewardComputer(RewardConfig(), executor, index, vocab)
    task = next(
        t for t in tasks.select(level=1)
        if tuple(t.call_names) in {o.call_names for o in vocab.level(1)} or t.call_names[0] == "read_csv"
    )
    programs = [task.reference, task.reference, "result = 1 / 0\n", ""]
    (level0,) = computer.score([(task, programs)], level=0)
    (level1,) = computer.score([(task, programs)], level=1)
    (level2,) = computer.score([(task, programs)], level=2)
    assert [r.exec for r in level0] == [1.0, 1.0, 0.3, 0.0]
    assert all(r.total == r.exec for r in level0)
    # Level 1 adds the quality term for programs that executed.
    assert level1[0].quality > 0 and level1[0].flags["no_redundancy"] and level1[0].flags["minimal_api"]
    assert level1[0].total == pytest.approx(1.0 + 0.2 * level1[0].quality)
    assert level1[2].total == 0.3 and level1[3].total == 0.0
    # Level 2 adds self-consistency: two of four samples agree on the output.
    assert level2[0].consistency == 0.5 and level2[2].consistency == 0.0
    assert level2[0].total == pytest.approx(1.0 + 0.2 * level2[0].quality + 0.1 * 0.5)
