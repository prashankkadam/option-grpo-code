"""End-to-end orchestration: Stage 0 preparation and construction of the components.

Stage 0 (``prepare``) turns SDK source code into the artefacts every later step reads:

    artifacts/<sdk>/index.json        static index of the public API
    artifacts/<sdk>/graphs.json       G_flow and G_sem (G_api is rebuilt from the index)
    artifacts/<sdk>/embeddings.npz    identifier embeddings
    artifacts/<sdk>/options.json      candidate options O_0^cand and O_1^cand
    artifacts/<sdk>/tasks.jsonl       curriculum tasks D_0, D_1, D_2 with property tests
    artifacts/<sdk>/stats.json        benchmark and discovery statistics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from option_grpo.config import Config
from option_grpo.execution.sandbox import Executor
from option_grpo.graphs.build import SDKGraphs, build_graphs
from option_grpo.options.discovery import discover_options
from option_grpo.options.option import OptionVocabulary
from option_grpo.sdk.introspect import SDKIndex, analyze_sdk
from option_grpo.sdk.registry import SDKSpec, get_sdk
from option_grpo.tasks.dataset import TaskSet
from option_grpo.tasks.generator import TaskGenerator
from option_grpo.tasks.validate import validate_tasks
from option_grpo.utils import get_logger, read_json, write_json

log = get_logger(__name__)


@dataclass
class Artifacts:
    spec: SDKSpec
    graphs: SDKGraphs
    vocab: OptionVocabulary
    tasks: TaskSet

    @property
    def index(self) -> SDKIndex:
        return self.graphs.index


def harness_sys_path(spec: SDKSpec, index: SDKIndex | None = None) -> list[str]:
    """Extra import path for the sandbox when the SDK is used from a source checkout."""
    root = spec.source_root or (index.root if index is not None else None)
    if not root:
        return []
    root_path = Path(root)
    parent = root_path.parent if root_path.is_dir() or root_path.suffix == ".py" else root_path
    if "site-packages" in parent.parts or "dist-packages" in parent.parts:
        return []
    return [str(parent)]


def build_executor(cfg: Config, spec: SDKSpec, index: SDKIndex) -> Executor:
    return Executor(index, cfg.execution, sys_path=harness_sys_path(spec, index))


def benchmark_stats(graphs: SDKGraphs, vocab: OptionVocabulary, tasks: TaskSet) -> dict[str, Any]:
    """The per-SDK rows of the benchmark and option-discovery tables."""
    counts = {level: len(tasks.select(level=level)) for level in (0, 1, 2)}
    return {
        "sdk": graphs.index.name,
        "version": graphs.index.version,
        "functions": graphs.flow.number_of_nodes(),
        "level0_tasks": counts[0],
        "level1_tasks": counts[1],
        "level2_tasks": counts[2],
        "property_tests": sum(len(t.tests) for t in tasks),
        "splits": tasks.counts(),
        "graphs": graphs.stats(),
        "options": vocab.stats(),
        "density": round(graphs.density(), 4),
    }


def prepare(cfg: Config, validate: bool = True, hints: dict[str, str] | None = None) -> Artifacts:
    """Stage 0 of Algorithm 4 for ``cfg.sdk``; writes and returns the artefacts."""
    spec = get_sdk(cfg.sdk)
    out = cfg.sdk_artifacts()
    out.mkdir(parents=True, exist_ok=True)

    index = analyze_sdk(
        spec.import_name,
        root=spec.source_root,
        name=spec.name,
        dist_name=spec.dist,
        exclude_dirs=cfg.graph.exclude_dirs,
        include_private=cfg.graph.include_private,
    )
    graphs = build_graphs(index, cfg.graph)
    graphs.save(out)

    vocab = discover_options(graphs, cfg.discovery, cfg.graph.min_compat)
    vocab.save(out / "options.json")

    generator = TaskGenerator(graphs, spec.import_prelude, cfg.benchmark, cfg.discovery, hints, cfg.graph.min_compat)
    tasks = generator.generate()
    report: list[dict[str, Any]] = []
    if validate:
        executor = build_executor(cfg, spec, index)
        try:
            tasks, report = validate_tasks(tasks, executor, cfg.benchmark)
        finally:
            executor.close()
    tasks.save(out / "tasks.jsonl")
    write_json(out / "skipped.json", {"synthesis": generator.skipped, "validation": report})
    stats = benchmark_stats(graphs, vocab, tasks)
    stats["validated"] = validate
    write_json(out / "stats.json", stats)
    log.info("prepared %s: %s", spec.name, {k: stats[k] for k in ("functions", "level0_tasks", "level1_tasks", "level2_tasks", "property_tests")})
    return Artifacts(spec, graphs, vocab, tasks)


def load_artifacts(cfg: Config, options_path: str | Path | None = None) -> Artifacts:
    """Load the Stage-0 artefacts of ``cfg.sdk``.

    ``options_path`` overrides the candidate vocabulary, e.g. with the certified
    vocabulary stored next to a trained checkpoint.
    """
    spec = get_sdk(cfg.sdk)
    directory = cfg.sdk_artifacts()
    if not (directory / "index.json").exists():
        raise FileNotFoundError(f"no artefacts in {directory}; run `option-grpo prepare --sdk {cfg.sdk}` first")
    graphs = SDKGraphs.load(directory)
    vocab = OptionVocabulary.load(options_path or directory / "options.json")
    tasks = TaskSet.load(directory / "tasks.jsonl")
    return Artifacts(spec, graphs, vocab, tasks)


def load_stats(cfg: Config) -> dict[str, Any]:
    return read_json(cfg.sdk_artifacts() / "stats.json")
