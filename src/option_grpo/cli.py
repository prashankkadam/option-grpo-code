"""Command-line interface.

    option-grpo prepare   --sdk daft            # Stage 0: graphs, options, tasks, tests
    option-grpo train     --sdk daft            # Algorithm 4: three curriculum phases
    option-grpo evaluate  --sdk daft --checkpoint runs/daft/checkpoints/final
    option-grpo baseline  --sdk daft --method rag
    option-grpo api       --sdk daft --model-name gpt-4o --model gpt-4o
    option-grpo probe     --sdk daft            # memorisation probe
    option-grpo analyze   odl runs/daft/eval    # ODL / quality correlations
    option-grpo stats     --sdk daft            # benchmark statistics of an SDK
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from option_grpo.config import Config, load_config
from option_grpo.utils import get_logger, read_jsonl, seed_everything, setup_logging, write_json

log = get_logger("option_grpo")


# --------------------------------------------------------------------------------------
# Shared argument handling
# --------------------------------------------------------------------------------------
def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", "-c", action="append", default=[], help="YAML config file (repeatable, later files win)")
    parser.add_argument("--override", "-o", action="append", default=[], help="dotted override, e.g. grpo.group_size=8")
    parser.add_argument("--sdk", help="SDK name from the registry")
    parser.add_argument("--sdk-registry", help="YAML file with additional SDK definitions")
    parser.add_argument("--output-dir", help="run directory (overrides output_dir)")
    parser.add_argument("--artifacts-dir", help="directory of the Stage-0 artefacts")


def _config(args: argparse.Namespace) -> Config:
    overrides = list(args.override)
    if getattr(args, "sdk", None):
        overrides.append(f"sdk={args.sdk}")
    if getattr(args, "output_dir", None):
        overrides.append(f"output_dir={args.output_dir}")
    if getattr(args, "artifacts_dir", None):
        overrides.append(f"artifacts_dir={args.artifacts_dir}")
    cfg = load_config(args.config or None, overrides)
    if getattr(args, "sdk_registry", None):
        from option_grpo.sdk.registry import load_registry

        load_registry(args.sdk_registry)
    setup_logging(cfg.logging.level)
    seed_everything(cfg.seed)
    return cfg


def _hints(path: str | None) -> dict[str, str] | None:
    if not path:
        return None
    import yaml

    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return {str(k): str(v) for k, v in data.items()}


def _print(obj: Any) -> None:
    json.dump(obj, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


# --------------------------------------------------------------------------------------
# Component construction shared by train / evaluate / baseline / probe
# --------------------------------------------------------------------------------------
def _runtime(
    cfg: Config,
    checkpoint: str | None,
    options_path: str | None = None,
    cpu: bool = False,
    checkpoint_options: bool = True,
):
    """Artefacts, policy, tokenizer and executor for a run.

    ``checkpoint_options`` makes the vocabulary stored with the checkpoint (including
    its certification status) take the place of the prepared candidates.  It is switched
    off when the checkpoint comes from another SDK.
    """
    from accelerate import Accelerator, DistributedDataParallelKwargs

    from option_grpo.pipeline import build_executor, load_artifacts
    from option_grpo.policy.hierarchical import build_policy

    if checkpoint and checkpoint_options and options_path is None and (Path(checkpoint) / "options.json").exists():
        options_path = str(Path(checkpoint) / "options.json")
    artifacts = load_artifacts(cfg, options_path)
    reserve = 64 if cfg.certification.allow_novel_patterns else 0
    tokenizer, policy = build_policy(cfg.model, len(artifacts.vocab) + reserve, cfg.decode.option_logit_bias, checkpoint)
    executor = build_executor(cfg, artifacts.spec, artifacts.index)
    accelerator = Accelerator(cpu=cpu, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    return artifacts, tokenizer, policy, executor, accelerator


def _retriever(cfg: Config, index, mode: str):
    if mode == "none":
        return None
    from option_grpo.eval.retrieval import Retriever, transformer_encoder

    return Retriever(index, transformer_encoder(cfg.eval.retriever_model), mode, cfg.eval.retrieval_top_k)


def _engine(cfg: Config, policy, tokenizer, artifacts):
    from option_grpo.options.constraints import ConstraintEngine, LanguageTables, VocabIndex
    from option_grpo.policy.sampling import RolloutEngine

    vocab_index = VocabIndex.from_tokenizer(tokenizer, policy.base_vocab_size)
    constraints = ConstraintEngine(
        vocab_index, LanguageTables.from_index(artifacts.index), artifacts.vocab.options,
        cfg.decode.strict_chain, cfg.decode.mask_cache_size,
    )
    return RolloutEngine(
        policy, tokenizer, artifacts.vocab, constraints, cfg.decode,
        stop_strings=cfg.model.stop_strings, stop_token_ids=cfg.model.extra_stop_token_ids,
    )


def _eval_tasks(cfg: Config, artifacts, levels: list[int] | None, limit: int | None):
    tasks = [t for t in artifacts.tasks.select(split=cfg.eval.split) if levels is None or t.level in levels]
    return tasks[:limit] if limit else tasks


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------
def cmd_prepare(args: argparse.Namespace) -> int:
    from option_grpo.pipeline import benchmark_stats, prepare

    cfg = _config(args)
    artifacts = prepare(cfg, validate=not args.no_validate, hints=_hints(args.hints))
    _print(benchmark_stats(artifacts.graphs, artifacts.vocab, artifacts.tasks))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from option_grpo.pipeline import load_stats

    _print(load_stats(_config(args)))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from option_grpo.grpo.trainer import OptionGRPOTrainer

    cfg = _config(args)
    checkpoint = args.resume or args.init_from
    artifacts, tokenizer, policy, executor, accelerator = _runtime(
        cfg, checkpoint, args.options, args.cpu, checkpoint_options=bool(args.resume)
    )
    retriever = _retriever(cfg, artifacts.index, args.retrieval)
    trainer = OptionGRPOTrainer(
        cfg, policy, tokenizer, artifacts.graphs, artifacts.vocab, artifacts.tasks, executor, accelerator,
        context_fn=retriever.context if retriever else None,
        init_options=checkpoint is None or bool(args.init_from),
    )
    if args.resume:
        trainer.load_state(args.resume)
    try:
        trainer.fit()
        if not args.no_eval and trainer.is_main:
            _evaluate_policy(cfg, trainer.engine, artifacts, executor, tokenizer, Path(cfg.output_dir) / "eval",
                             hierarchical=cfg.curriculum.use_options, context_fn=trainer.context_fn, limit=args.eval_limit)
    finally:
        executor.close()
    return 0


def _evaluate_policy(cfg, engine, artifacts, executor, tokenizer, out_dir, hierarchical, context_fn=None, levels=None, limit=None):
    from option_grpo.analysis.quality import tokenizer_offsets
    from option_grpo.eval.evaluator import Evaluator, PolicyGenerator
    from option_grpo.options.tracker import InitiationTable

    initiation = None
    if hierarchical:
        table = InitiationTable(artifacts.vocab, artifacts.index, 2, cfg.decode.initiation, cfg.decode.min_initiation_compat)
        initiation = table if table.available else None
    generator = PolicyGenerator(engine, initiation, context_fn, cfg.grpo.rollout_chunk)
    evaluator = Evaluator(
        artifacts.index, artifacts.vocab, executor, cfg.eval, tokenizer_offsets(tokenizer), cfg.benchmark.odl_max_options
    )
    report = evaluator.evaluate(_eval_tasks(cfg, artifacts, levels, limit), generator, artifacts.spec.name)
    report.save(out_dir)
    _print(report.to_json())
    return report


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = _config(args)
    artifacts, tokenizer, policy, executor, _ = _runtime(cfg, args.checkpoint, args.options, args.cpu)
    try:
        import torch

        if torch.cuda.is_available() and not args.cpu:
            policy.to("cuda")
        policy.eval()
        engine = _engine(cfg, policy, tokenizer, artifacts)
        retrieval = args.retrieval if args.retrieval != "none" else cfg.eval.retrieval
        retriever = _retriever(cfg, artifacts.index, retrieval)
        out_dir = Path(args.output_dir or cfg.output_dir) / (args.name or "eval")
        _evaluate_policy(cfg, engine, artifacts, executor, tokenizer, out_dir, hierarchical=not args.flat,
                         context_fn=retriever.context if retriever else None, levels=args.levels, limit=args.limit)
    finally:
        executor.close()
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    """Frozen model, optionally with retrieval (RAG / DocPrompting)."""
    cfg = _config(args)
    artifacts, tokenizer, policy, executor, _ = _runtime(cfg, None, args.options, args.cpu)
    try:
        import torch

        if torch.cuda.is_available() and not args.cpu:
            policy.to("cuda")
        policy.eval()
        engine = _engine(cfg, policy, tokenizer, artifacts)
        mode = {"frozen": "none", "rag": "rag", "docprompting": "docprompting"}[args.method]
        retriever = _retriever(cfg, artifacts.index, mode)
        out_dir = Path(args.output_dir or cfg.output_dir) / f"baseline-{args.method}"
        _evaluate_policy(cfg, engine, artifacts, executor, tokenizer, out_dir, hierarchical=False,
                         context_fn=retriever.context if retriever else None, levels=args.levels, limit=args.limit)
    finally:
        executor.close()
    return 0


def cmd_api(args: argparse.Namespace) -> int:
    """Evaluate a hosted model through an OpenAI-compatible endpoint."""
    from option_grpo.eval.backends import ApiModel, ChatCompletionsGenerator
    from option_grpo.eval.evaluator import Evaluator
    from option_grpo.pipeline import build_executor, load_artifacts

    cfg = _config(args)
    artifacts = load_artifacts(cfg, args.options)
    executor = build_executor(cfg, artifacts.spec, artifacts.index)
    try:
        retriever = _retriever(cfg, artifacts.index, args.retrieval)
        spec = ApiModel(args.model_name, args.model, args.base_url, args.api_key_env, concurrency=args.concurrency)
        generator = ChatCompletionsGenerator(spec, retriever.context if retriever else None)
        evaluator = Evaluator(artifacts.index, artifacts.vocab, executor, cfg.eval, max_options=cfg.benchmark.odl_max_options)
        report = evaluator.evaluate(_eval_tasks(cfg, artifacts, args.levels, args.limit), generator, artifacts.spec.name)
        report.save(Path(args.output_dir or cfg.output_dir) / f"api-{args.model_name}")
        _print(report.to_json())
    finally:
        executor.close()
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    from option_grpo.eval.probe import PolicyCompleter, build_probes, run_probe
    from option_grpo.pipeline import load_artifacts
    from option_grpo.policy.hierarchical import build_policy
    from option_grpo.policy.sampling import RolloutEngine

    cfg = _config(args)
    artifacts = load_artifacts(cfg)
    tokenizer, policy = build_policy(cfg.model, 0)
    import torch

    if torch.cuda.is_available() and not args.cpu:
        policy.to("cuda")
    policy.eval()
    engine = RolloutEngine(policy, tokenizer, artifacts.vocab, None, cfg.decode, stop_strings=["\n"])
    probes = build_probes(artifacts.index, artifacts.spec.title, args.n, cfg.seed)
    result = run_probe(probes, PolicyCompleter(engine))
    out = Path(args.output_dir or cfg.output_dir) / "probe.json"
    write_json(out, result)
    _print({"sdk": artifacts.spec.name, "n": result["n"], "recall": result["recall"], "by_kind": result["by_kind"]})
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    setup_logging("INFO")
    if args.what == "odl":
        from option_grpo.eval.evaluator import odl_quality_correlations

        records = list(read_jsonl(Path(args.path) / "samples.jsonl"))
        _print(odl_quality_correlations(records))
    elif args.what == "gradvar":
        from option_grpo.grpo.trainer import relative_gradient_variance

        _print(relative_gradient_variance(Path(args.path) / "metrics.jsonl", args.window))
    elif args.what == "curve":
        rows = list(read_jsonl(Path(args.path) / "metrics.jsonl"))
        window = max(args.window, 1)
        curve = []
        for start in range(0, len(rows), window):
            chunk = rows[start : start + window]
            curve.append({
                "global_step": chunk[-1]["global_step"],
                "phase": chunk[-1]["phase"],
                "pass_rate": sum(r["pass_rate"] for r in chunk) / len(chunk),
                "reward": sum(r["reward"] for r in chunk) / len(chunk),
            })
        _print(curve)
    return 0


# --------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="option-grpo", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Stage 0: build graphs, discover options, generate tasks and tests")
    _add_common(p)
    p.add_argument("--no-validate", action="store_true", help="skip executing the reference skeletons")
    p.add_argument("--hints", help="YAML file mapping 'function.param' or 'param' to literal values")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("stats", help="print the benchmark statistics of a prepared SDK")
    _add_common(p)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("train", help="curriculum-staged Option-GRPO training")
    _add_common(p)
    p.add_argument("--resume", help="checkpoint directory to resume from")
    p.add_argument("--init-from", help="checkpoint used as initialisation (cross-SDK transfer)")
    p.add_argument("--options", help="option vocabulary to use instead of the prepared candidates")
    p.add_argument("--retrieval", choices=["none", "rag", "docprompting"], default="none")
    p.add_argument("--no-eval", action="store_true", help="skip the evaluation after training")
    p.add_argument("--eval-limit", type=int, help="evaluate on at most this many tasks")
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=cmd_train)

    def add_eval_args(q: argparse.ArgumentParser) -> None:
        _add_common(q)
        q.add_argument("--options", help="option vocabulary used for decoding and quality metrics")
        q.add_argument("--levels", type=int, nargs="*", help="curriculum levels to evaluate (default: all)")
        q.add_argument("--limit", type=int, help="evaluate on at most this many tasks")
        q.add_argument("--cpu", action="store_true")

    p = sub.add_parser("evaluate", help="evaluate a trained checkpoint")
    add_eval_args(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--flat", action="store_true", help="token-level decoding (checkpoints trained without options)")
    p.add_argument("--retrieval", choices=["none", "rag", "docprompting"], default="none")
    p.add_argument("--name", help="sub-directory for the results")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("baseline", help="evaluate the frozen base model, optionally with retrieval")
    add_eval_args(p)
    p.add_argument("--method", choices=["frozen", "rag", "docprompting"], default="frozen")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("api", help="evaluate a hosted model via an OpenAI-compatible endpoint")
    add_eval_args(p)
    p.add_argument("--model-name", required=True, help="label used in the report")
    p.add_argument("--model", required=True, help="model identifier sent to the endpoint")
    p.add_argument("--base-url", default="https://api.openai.com/v1")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--retrieval", choices=["none", "rag", "docprompting"], default="rag")
    p.set_defaults(func=cmd_api)

    p = sub.add_parser("probe", help="memorisation probe of the frozen base model")
    _add_common(p)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("analyze", help="post-hoc analyses of a run")
    p.add_argument("what", choices=["odl", "gradvar", "curve"])
    p.add_argument("path", help="evaluation directory (odl) or run directory (gradvar, curve)")
    p.add_argument("--window", type=int, default=200)
    p.set_defaults(func=cmd_analyze)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
