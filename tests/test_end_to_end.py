"""The whole pipeline on the bundled example SDK with a tiny language model (CPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from option_grpo.analysis.quality import tokenizer_offsets
from option_grpo.cli import main as cli_main
from option_grpo.config import load_config
from option_grpo.eval.evaluator import Evaluator, PolicyGenerator, odl_quality_correlations
from option_grpo.grpo.trainer import OptionGRPOTrainer, relative_gradient_variance
from option_grpo.options.option import OptionVocabulary
from option_grpo.policy.hierarchical import HierarchicalPolicy
from option_grpo.utils import read_jsonl

REPO = Path(__file__).resolve().parents[1]


def test_config_overrides_and_validation(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"grpo": {"group_size": 8}, "optim": {"lr": "1e-4"}}))
    cfg = load_config(path, ["curriculum.steps=[1, 2, 3]", "model.name_or_path=some/model"])
    assert cfg.grpo.group_size == 8 and cfg.optim.lr == 1e-4 and cfg.curriculum.steps == [1, 2, 3]
    assert cfg.grpo.kl_beta == 0.04 and cfg.grpo.option_loss_weight == 0.3 and cfg.reward.quality_weight == 0.2
    with pytest.raises(KeyError):
        load_config(None, ["grpo.no_such_key=1"])
    with pytest.raises(ValueError):
        load_config(None, ["grpo.group_size=1"])


def test_prepare_command_writes_stage0_artifacts(tmp_path, capsys):
    out = tmp_path / "artifacts"
    code = cli_main([
        "prepare", "--sdk", "tinyframe", "--sdk-registry", str(REPO / "examples" / "tinyframe" / "sdk.yaml"),
        "--artifacts-dir", str(out), "-o", "graph.embedder=hashing",
    ])
    assert code == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["functions"] == 49 and stats["level0_tasks"] > 30 and stats["property_tests"] > 0
    for name in ("index.json", "graphs.json", "embeddings.npz", "options.json", "tasks.jsonl", "stats.json", "skipped.json"):
        assert (out / "tinyframe" / name).exists()
    vocab = OptionVocabulary.load(out / "tinyframe" / "options.json")
    assert vocab.stats()["level0"] == 49 and vocab.stats()["level1_candidates"] > 0


@pytest.mark.slow
def test_three_phase_training_and_evaluation(cfg, lm, tokenizer, graphs, vocab, tasks, executor):
    from accelerate import Accelerator

    torch.manual_seed(0)
    policy = HierarchicalPolicy(lm, len(vocab))
    trainer = OptionGRPOTrainer(cfg, policy, tokenizer, graphs, vocab, tasks, executor, Accelerator(cpu=True))
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    trainer.fit()

    run = Path(cfg.output_dir)
    rows = list(read_jsonl(run / "metrics.jsonl"))
    assert [r["phase"] for r in rows] == [1, 1, 2, 2, 3, 3]
    assert all(0.0 <= r["reward_exec"] <= 1.0 and r["tokens"] > 0 for r in rows)
    assert any(r["updated"] for r in rows)
    assert any(not torch.equal(before[name], p.detach()) for name, p in policy.named_parameters())
    for name in ("phase1", "phase2", "phase3", "final"):
        checkpoint = run / "checkpoints" / name
        assert (checkpoint / "options.pt").exists() and (checkpoint / "options.json").exists()
        assert (checkpoint / "state.json").exists() and (checkpoint / "lm").is_dir()
    assert [c["level"] for c in trainer.state.certification] == [0, 1]
    # The tiny model was fitted to the reference programs, so it solves Level-0 tasks and
    # the corresponding atomic options are certified and enter the next action space.
    assert trainer.state.certification[0]["certified"] > 0
    assert trainer.initiation_for(1) is not None
    assert isinstance(relative_gradient_variance(run / "metrics.jsonl", window=2), list)

    evaluator = Evaluator(graphs.index, vocab, executor, cfg.eval, tokenizer_offsets(tokenizer))
    generator = PolicyGenerator(trainer.engine, trainer.initiation_for(2), chunk=2)
    report = evaluator.evaluate(tasks.select(level=0, split="test")[:6], generator)
    assert report.n_tasks == 6 and len(report.records) == 12
    assert 0.0 <= report.metrics["pass@1"] <= 100.0 and report.metrics["pass@2"] >= report.metrics["pass@1"]
    report.save(run / "eval")
    samples = list(read_jsonl(run / "eval" / "samples.jsonl"))
    assert set(odl_quality_correlations(samples)) == {"conciseness", "api_efficiency", "redundancy", "idiomaticity"}

    # A checkpoint restores the option head and the curriculum state.
    restored = HierarchicalPolicy(type(lm).from_pretrained(run / "checkpoints" / "final" / "lm"), len(vocab))
    restored.load_option_state(torch.load(run / "checkpoints" / "final" / "options.pt"))
    assert torch.allclose(restored.option_in.weight, policy.option_in.weight)
    assert torch.allclose(restored.token_embeddings.weight, policy.token_embeddings.weight)


@pytest.mark.slow
def test_flat_grpo_ablation_runs_a_single_token_level_phase(cfg, lm, tokenizer, graphs, vocab, tasks, executor):
    from accelerate import Accelerator

    cfg.curriculum.use_options = False
    cfg.curriculum.use_curriculum = False
    cfg.curriculum.steps = [1, 1, 1]
    cfg.grpo.reference = "none"
    trainer = OptionGRPOTrainer(cfg, HierarchicalPolicy(lm, 0), tokenizer, graphs, vocab, tasks, executor, Accelerator(cpu=True))
    trainer.fit()
    rows = list(read_jsonl(Path(cfg.output_dir) / "metrics.jsonl"))
    assert len(rows) == 3 and {r["phase"] for r in rows} == {1}
    assert all(r["option_decisions"] == 0 and r["kl"] == 0.0 for r in rows)
    assert trainer.state.certification == []
