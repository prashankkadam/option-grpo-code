"""Task validation.

Every generated task carries a call skeleton synthesised from its path.  Validation
executes that skeleton in the sandbox and

* discards tasks whose skeleton does not run or does not produce the expected type
  (the task would be unsolvable as stated);
* records how the skeleton behaves on each edge-case variant, turning the generic
  edge-case tests into precise expectations (same output, or the same exception);
* removes property tests that the skeleton itself violates (for instance a determinism
  test on a function that is random by design);
* marks the function-level properties the skeleton exercises as *required*, so that a
  program cannot satisfy them vacuously by never calling the function;
* optionally adds a differential test against the skeleton's output.
"""

from __future__ import annotations

from typing import Any

from option_grpo.config import BenchmarkConfig
from option_grpo.execution.sandbox import Executor, build_job
from option_grpo.tasks.dataset import Task, TaskSet
from option_grpo.utils import get_logger

log = get_logger(__name__)

_RAN = {"output_none", "type_mismatch", "type_ok", "tests_passed"}
_CALL_LEVEL = {"call_type", "shape", "invariant"}


def validate_tasks(
    tasks: TaskSet,
    executor: Executor,
    cfg: BenchmarkConfig | None = None,
) -> tuple[TaskSet, list[dict[str, Any]]]:
    """Return the validated task set and a report of dropped tasks and tests."""
    cfg = cfg or BenchmarkConfig()
    sandbox = executor.sandbox
    report: list[dict[str, Any]] = []
    candidates = [t for t in tasks if t.reference]

    def job(task: Task, **kwargs: Any) -> dict[str, Any]:
        return build_job(task, task.reference or "", executor.index, executor.cfg, executor.sys_path, **kwargs)

    # 1. The skeleton must run and produce the expected type.
    base_results = sandbox.run_many([job(t, tests=[]) for t in candidates])
    alive: list[Task] = []
    digests: dict[str, str | None] = {}
    for task, result in zip(candidates, base_results):
        if result.stage not in {"type_ok", "tests_passed"}:
            report.append({"task": task.id, "dropped": True, "stage": result.stage, "error": result.error})
            continue
        digests[task.id] = result.output_digest
        alive.append(task)

    # 2. Behaviour of the skeleton on every edge-case variant.
    variant_jobs = []
    variant_keys: list[tuple[Task, dict[str, Any]]] = []
    for task in alive:
        for variant in task.variants:
            variant_jobs.append(job(task, tests=[], setup=variant["setup"], fixtures=variant.get("fixtures")))
            variant_keys.append((task, variant))
    expectations: dict[tuple[str, str], Any] = {}
    for (task, variant), result in zip(variant_keys, sandbox.run_many(variant_jobs)):
        if result.stage in _RAN:
            # With differential testing enabled the variant must reproduce the skeleton's
            # output; otherwise it only has to run and return the expected type.
            exact = cfg.reference_equivalence and result.output_digest
            expectations[(task.id, variant["name"])] = {"digest": result.output_digest} if exact else "ok"
        elif result.stage == "runtime_error" and result.error:
            expectations[(task.id, variant["name"])] = {"raises": result.error["type"]}
        else:
            expectations[(task.id, variant["name"])] = None  # unusable variant
    for task in alive:
        kept_variants = []
        for variant in task.variants:
            if expectations.get((task.id, variant["name"])) is not None:
                kept_variants.append(variant)
        task.variants = kept_variants
        kept_tests = []
        for test in task.tests:
            if test["kind"] == "edge":
                expect = expectations.get((task.id, test["variant"]))
                if expect is None:
                    continue
                test = dict(test, expect=expect)
            kept_tests.append(test)
        if cfg.reference_equivalence and digests.get(task.id):
            kept_tests.append({"kind": "reference", "id": f"reference-{len(kept_tests)}", "digest": digests[task.id]})
        task.tests = kept_tests

    # 3. A property the skeleton violates is not a property of the task.
    final_results = sandbox.run_many([job(t) for t in alive])
    validated: list[Task] = []
    for task, result in zip(alive, final_results):
        if result.stage not in {"type_ok", "tests_passed"}:
            report.append({"task": task.id, "dropped": True, "stage": result.stage, "error": result.error})
            continue
        failing = {t["id"] for t in result.tests if not t.get("passed")}
        if failing:
            report.append({"task": task.id, "dropped": False, "removed_tests": sorted(failing)})
            task.tests = [t for t in task.tests if t.get("id") not in failing]
        if cfg.require_calls:
            # A property of a path function only counts when the program calls that
            # function.  It is required exactly when the skeleton exercises it, which
            # also guarantees that the call is observable by the tracer.
            exercised = {t["id"] for t in result.tests if t.get("passed") and not t.get("skipped")}
            task.tests = [
                dict(t, required=True) if t["kind"] in _CALL_LEVEL and t.get("id") in exercised else t
                for t in task.tests
            ]
        task.validated = True
        validated.append(task)
    dropped = sum(1 for r in report if r.get("dropped"))
    log.info("validated %d of %d tasks (%d dropped)", len(validated), len(candidates), dropped)
    return TaskSet(validated), report
