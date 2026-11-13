"""The hierarchical policy, constrained rollouts and option extraction."""

from __future__ import annotations

import pytest
import torch

from option_grpo.analysis.callseq import analyze_program
from option_grpo.config import FreezeConfig, OptimConfig, OptionDecodeConfig
from option_grpo.grpo.optim import apply_partial_freeze, build_optimizer
from option_grpo.options.constraints import ConstraintEngine, LanguageTables, VocabIndex
from option_grpo.options.extraction import annotate_trajectory
from option_grpo.options.option import CERTIFIED
from option_grpo.options.tracker import InitiationState, InitiationTable
from option_grpo.policy.hierarchical import HierarchicalPolicy
from option_grpo.policy.sampling import DECISION, GLUE, INTRA, RolloutEngine
from option_grpo.tasks.prompts import END_MARKER, build_prompt


@pytest.fixture()
def setup(lm, tokenizer, vocab, index):
    for option in vocab.options:
        option.status = CERTIFIED
    torch.manual_seed(0)
    policy = HierarchicalPolicy(lm, len(vocab), option_logit_bias=8.0)
    policy.init_options(vocab.options, tokenizer)
    constraints = ConstraintEngine(VocabIndex.from_tokenizer(tokenizer, policy.base_vocab_size), LanguageTables.from_index(index), vocab.options)
    engine = RolloutEngine(policy, tokenizer, vocab, constraints, OptionDecodeConfig(max_option_tokens=40), stop_strings=[END_MARKER])
    return policy, engine, constraints


def test_extended_action_space_shapes(setup, tokenizer, vocab):
    policy, _, _ = setup
    assert policy.base_vocab_size == len(tokenizer)
    ids = torch.tensor([[1, 2, policy.base_vocab_size + 3]])
    embeds = policy.embed(ids)
    assert embeds.shape == (1, 3, policy.hidden_size)
    assert torch.allclose(embeds[0, 2], policy.option_in.weight[3])
    token_logits, option_logits = policy(ids, torch.ones_like(ids), torch.tensor([0, 0]), torch.tensor([1, 2]))
    assert token_logits.shape == (2, len(tokenizer)) and option_logits.shape == (2, len(vocab))


def test_initiation_sets_follow_types(vocab, index):
    for option in vocab.options:
        option.status = CERTIFIED
    table = InitiationTable(vocab, index, level=1)
    read_csv, where = vocab.get("L0:tinyframe.read_csv"), vocab.get("L0:tinyframe.Frame.where")
    to_list = vocab.get("L0:tinyframe.Query.to_list")
    assert all(vocab[i].level == 0 for i in table.available)  # A_1 holds atomic options only
    state = InitiationState(table)
    assert read_csv.index in state.allowed and where.index not in state.allowed  # nothing to filter yet
    state.on_terminate(read_csv, completed=True)
    assert where.index in state.allowed and to_list.index not in state.allowed
    assert read_csv.index in state.allowed  # source options are always initiable
    before = state.allowed
    state.on_terminate(where, completed=False)  # an aborted option produced no value
    assert state.allowed == before
    assert len(InitiationTable(vocab, index, level=2).available) > len(table.available)
    assert InitiationTable(vocab, index, level=0).available == ()


def test_rollouts_respect_options_and_are_reproducible_under_the_policy(setup, tasks, vocab, index):
    policy, engine, constraints = setup
    table = InitiationTable(vocab, index, level=2)
    task = next(t for t in tasks.select(level=1) if t.call_names[0] == "read_csv")
    trajectories = []
    for seed in range(3):
        torch.manual_seed(seed)
        trajectories += engine.rollout(task.id, build_prompt(task), 6, table, task.setup_value_types(), 0.8, 48)
    assert sum(len(t.calls) for t in trajectories) > 0, "the biased policy should initiate options"
    strings = constraints.vocab.strings
    completed = 0
    for traj in trajectories:
        assert len(traj.actions) == len(traj.logp) == len(traj.kinds) == len(traj.masks)
        for call in traj.calls:
            assert traj.kinds[call.start] == DECISION
            assert traj.actions[call.start] == policy.base_vocab_size + call.option
            assert all(kind == INTRA for kind in traj.kinds[call.start + 1 : call.end])
            segment = "".join(strings[a] for a in traj.actions[call.start + 1 : call.end])
            if call.completed:
                completed += 1
                # A completed option contains its call path, in order, on the main chain.
                position = 0
                for name in vocab[call.option].call_names:
                    position = segment.index(name + "(", position) + 1
        outside = [k for i, k in enumerate(traj.kinds) if not any(c.start <= i < c.end for c in traj.calls)]
        assert all(kind == GLUE for kind in outside)
    assert completed > 0
    # Re-scoring under the same parameters reproduces the recorded log-probabilities,
    # which shows that the masks stored per step describe the sampling distribution exactly.
    with torch.no_grad():
        for traj, logp in zip(trajectories, engine.log_probs(trajectories)):
            assert torch.allclose(logp, torch.tensor(traj.logp), atol=1e-4)


def test_token_level_rollout_without_options(setup, tasks):
    _, engine, _ = setup
    task = tasks.select(level=0)[0]
    torch.manual_seed(0)
    trajectories = engine.rollout(task.id, build_prompt(task), 3, None, None, 0.8, 24)
    assert all(not t.calls and set(t.kinds) <= {GLUE} for t in trajectories)
    greedy_a = engine.rollout(task.id, build_prompt(task), 1, None, None, 0.0, 24)[0]
    greedy_b = engine.rollout(task.id, build_prompt(task), 1, None, None, 0.0, 24)[0]
    assert greedy_a.actions == greedy_b.actions
    assert all(END_MARKER not in t.text for t in trajectories)


def test_extraction_rewrites_a_solution_as_an_option_trajectory(setup, tasks, tokenizer, vocab, index):
    policy, engine, constraints = setup
    task = next(t for t in tasks.select(level=1) if t.call_names[0] == "read_csv" and "\n" not in t.reference.strip())
    table = InitiationTable(vocab, index, level=1)  # atomic options only
    prompt_ids = engine.encode_prompt(build_prompt(task))
    token_ids = tokenizer(task.reference + END_MARKER)["input_ids"]
    plain = engine.rollout(task.id, build_prompt(task), 1, None, None, 0.0, 1)[0]
    plain.prompt_ids, plain.actions = prompt_ids, token_ids
    plain.kinds, plain.logp, plain.masks = [GLUE] * len(token_ids), [0.0] * len(token_ids), [None] * len(token_ids)
    plain.text, plain.calls = task.reference, []
    rewritten = annotate_trajectory(
        plain, index, vocab, constraints, table, policy.base_vocab_size, engine.stop_token_ids,
        task.context_code(), task.setup_value_types(),
    )
    assert rewritten is not None and rewritten.calls
    # The token content is unchanged; only option decisions were inserted.
    assert [a for a, k in zip(rewritten.actions, rewritten.kinds) if k != DECISION] == token_ids
    used = [vocab[c.option].call_names[0] for c in rewritten.calls]
    primary = analyze_program(task.reference, index, task.context_code()).primary_names()
    remaining = iter(primary)
    assert used[0] == primary[0] and all(name in remaining for name in used)  # an ordered subsequence
    assert all(c.completed for c in rewritten.calls)
    with torch.no_grad():
        (logp,) = engine.log_probs([rewritten])
    assert torch.isfinite(logp).all()  # every rewritten action is admissible under its mask


def test_parameter_groups_and_partial_freeze(setup):
    policy, _, _ = setup
    optim_cfg = OptimConfig()
    optimizer = build_optimizer(policy, optim_cfg)
    seen = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(seen) == len(set(seen)) == len({id(p) for p in policy.parameters()})
    roles = {group["role"] for group in optimizer.param_groups}
    assert {"option", "embedding", "layer"} <= roles
    cutoff = apply_partial_freeze(optimizer, policy, 2, FreezeConfig(), optim_cfg)
    assert cutoff == 1  # half of the two blocks after the second phase
    for group in optimizer.param_groups:
        if group["role"] == "layer":
            assert group["lr_scale"] == pytest.approx(0.1 if group["layer"] < cutoff else 1.0)
        if group["role"] == "option":
            assert group["lr_scale"] == 1.0


@pytest.mark.parametrize("family", ["qwen", "llama"])
def test_policy_runs_on_rotary_decoder_architectures(family, tokenizer, vocab, index, tasks):
    """The wrapper relies only on the generic causal-LM interface (decoder, head, KV cache)."""
    import transformers

    if family == "qwen":
        names = [("Qwen3Config", "Qwen3ForCausalLM"), ("Qwen2Config", "Qwen2ForCausalLM")]
    else:
        names = [("LlamaConfig", "LlamaForCausalLM")]
    pair = next(((c, m) for c, m in names if hasattr(transformers, c) and hasattr(transformers, m)), None)
    if pair is None:
        pytest.skip(f"no {family} architecture in this transformers version")
    config = getattr(transformers, pair[0])(
        vocab_size=len(tokenizer) + 5, hidden_size=48, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
    )
    torch.manual_seed(0)
    lm = getattr(transformers, pair[1])(config)
    for option in vocab.options:
        option.status = CERTIFIED
    policy = HierarchicalPolicy(lm, len(vocab), option_logit_bias=8.0)
    policy.init_options(vocab.options, tokenizer)
    assert policy.base_vocab_size == len(tokenizer) + 5 and len(policy.transformer_layers()) == 2
    # Embedding rows without a token (ids beyond the tokenizer) are never sampled inside options.
    vocab_index = VocabIndex.from_tokenizer(tokenizer, policy.base_vocab_size)
    assert set(range(len(tokenizer), len(tokenizer) + 5)) <= set(vocab_index.always_banned)
    constraints = ConstraintEngine(vocab_index, LanguageTables.from_index(index), vocab.options)
    engine = RolloutEngine(policy, tokenizer, vocab, constraints, OptionDecodeConfig(max_option_tokens=24), stop_strings=[END_MARKER])
    task = tasks.select(level=1)[0]
    trajectories = engine.rollout(task.id, build_prompt(task), 4, InitiationTable(vocab, index, 2), task.setup_value_types(), 0.8, 32)
    assert any(t.calls for t in trajectories)
    logp = engine.log_probs(trajectories)
    for traj, values in zip(trajectories, logp):
        assert torch.allclose(values.detach(), torch.tensor(traj.logp), atol=1e-4)
    (-sum(v.sum() for v in logp)).backward()
    assert policy.option_out.weight.grad is not None and policy.option_in.weight.grad.abs().sum() > 0
