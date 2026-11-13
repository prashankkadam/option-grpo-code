"""Shared fixtures: the bundled example SDK, a tiny tokenizer and a tiny language model.

Everything here runs offline and on CPU in a few seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "tinyframe"
if str(EXAMPLE) not in sys.path:
    sys.path.insert(0, str(EXAMPLE))

from option_grpo.config import Config, ExecutionConfig, GraphConfig  # noqa: E402
from option_grpo.execution.sandbox import Executor  # noqa: E402
from option_grpo.graphs.build import build_graphs  # noqa: E402
from option_grpo.options.discovery import discover_options  # noqa: E402
from option_grpo.sdk.introspect import analyze_sdk  # noqa: E402
from option_grpo.tasks.generator import TaskGenerator  # noqa: E402
from option_grpo.tasks.prompts import END_MARKER, build_prompt  # noqa: E402
from option_grpo.tasks.validate import validate_tasks  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_logging():
    """Commands install their own log handlers; keep them from leaking between tests."""
    import logging

    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.fixture(scope="session")
def index():
    return analyze_sdk("tinyframe", root=EXAMPLE / "tinyframe")


@pytest.fixture(scope="session")
def graphs(index):
    return build_graphs(index, GraphConfig(embedder="hashing"))


@pytest.fixture()
def vocab(graphs):
    """A fresh candidate vocabulary (tests mutate certification status)."""
    return discover_options(graphs)


@pytest.fixture(scope="session")
def executor(index):
    ex = Executor(index, ExecutionConfig(workers=8, timeout_s=30), sys_path=[str(EXAMPLE)])
    yield ex
    ex.close()


@pytest.fixture(scope="session")
def tasks(graphs, executor):
    generated = TaskGenerator(graphs, "import tinyframe").generate()
    validated, _ = validate_tasks(generated, executor)
    return validated


@pytest.fixture(scope="session")
def tokenizer(tasks):
    """A small byte-level BPE tokenizer trained on the benchmark's own programs."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    corpus = []
    for task in tasks:
        corpus.append(build_prompt(task) + (task.reference or "") + END_MARKER + "\n")
    for path in sorted((EXAMPLE / "tinyframe").glob("*.py")):
        corpus.append(path.read_text())
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=700,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(corpus, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", pad_token="<|endoftext|>")


def make_lm(tokenizer, seed: int = 0):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=len(tokenizer), n_positions=1024, n_embd=64, n_layer=2, n_head=2,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    return GPT2LMHeadModel(config)


@pytest.fixture(scope="session")
def pretrained_state(tokenizer, tasks):
    """Weights of a tiny LM fitted to the reference programs (a stand-in for a code LM)."""
    lm = make_lm(tokenizer)
    texts = [build_prompt(t) + (t.reference or "") + END_MARKER + "\n" for t in tasks]
    encoded = [torch.tensor(tokenizer(t)["input_ids"][:1024]) for t in texts]
    optimizer = torch.optim.AdamW(lm.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(0)
    lm.train()
    for _ in range(700):
        picks = torch.randint(len(encoded), (8,), generator=generator).tolist()
        width = max(len(encoded[i]) for i in picks)
        batch = torch.full((len(picks), width), tokenizer.pad_token_id)
        labels = torch.full((len(picks), width), -100)
        for row, i in enumerate(picks):
            batch[row, : len(encoded[i])] = encoded[i]
            labels[row, : len(encoded[i])] = encoded[i]
        loss = lm(input_ids=batch, attention_mask=(labels != -100).long(), labels=labels).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return {k: v.clone() for k, v in lm.state_dict().items()}


@pytest.fixture()
def lm(tokenizer, pretrained_state):
    model = make_lm(tokenizer)
    model.load_state_dict(pretrained_state)
    return model


@pytest.fixture()
def cfg(tmp_path):
    config = Config()
    config.sdk = "tinyframe"
    config.output_dir = str(tmp_path / "run")
    config.artifacts_dir = str(tmp_path / "artifacts")
    config.graph.embedder = "hashing"
    config.model.dtype = "fp32"
    config.model.gradient_checkpointing = False
    config.grpo.group_size = 4
    config.grpo.max_new_tokens = 48
    config.grpo.rollout_chunk = 4
    config.grpo.train_micro_batch = 2
    config.optim.lr = 1e-3
    config.optim.warmup_steps = 2
    config.curriculum.steps = [2, 2, 2]
    config.curriculum.advance_window = 1
    config.certification.samples_per_task = 2
    config.certification.frequency_threshold = 1
    config.certification.max_tasks = 6
    config.execution.workers = 8
    config.logging.log_every = 1
    config.logging.save_every = 0
    config.eval.n_samples = 2
    config.eval.ks = [1, 2]
    config.eval.max_new_tokens = 48
    return config
