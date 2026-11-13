"""Advantages, the clipped objective, schedules, phases and small statistics."""

from __future__ import annotations

import math

import pytest
import torch

from option_grpo.config import CurriculumConfig, OptimConfig
from option_grpo.eval.stats import pass_at_k, spearman
from option_grpo.grpo.advantages import option_advantages, sequence_advantages
from option_grpo.grpo.curriculum import LengthPacer, build_phases
from option_grpo.grpo.loss import grpo_loss
from option_grpo.grpo.optim import schedule_factor
from option_grpo.policy.sampling import DECISION, GLUE


def test_sequence_advantages_are_group_normalised():
    adv = sequence_advantages(torch.tensor([1.0, 0.0, 0.0, 1.0]), eps=0.0)
    assert torch.allclose(adv, torch.tensor([1.0, -1.0, -1.0, 1.0]))
    assert torch.equal(sequence_advantages(torch.tensor([0.5, 0.5, 0.5])), torch.zeros(3))
    assert torch.equal(sequence_advantages(torch.tensor([0.7])), torch.zeros(1))


def test_option_advantage_is_mean_with_minus_mean_without():
    rewards = [1.0, 0.8, 0.2, 0.0]
    sets = [{1, 2}, {1}, {2}, set()]
    adv = option_advantages(rewards, sets)
    assert adv[1] == pytest.approx((1.0 + 0.8) / 2 - (0.2 + 0.0) / 2)
    assert adv[2] == pytest.approx((1.0 + 0.2) / 2 - (0.8 + 0.0) / 2)
    # An option used by every sample carries no group-relative signal.
    assert option_advantages([1.0, 0.0], [{3}, {3}]) == {3: 0.0}
    assert option_advantages([1.0, 0.0], [set(), set()]) == {}


def _loss(new, old, kinds, seq_adv, opt_adv, **kwargs):
    return grpo_loss(
        logp_new=[new], logp_old=[old], kinds=[kinds], seq_adv=torch.tensor([seq_adv]), opt_adv=[opt_adv], **kwargs
    )


def test_loss_gradient_raises_probability_of_advantaged_actions():
    new = torch.zeros(3, requires_grad=True)
    kinds = torch.tensor([DECISION, GLUE, GLUE])
    terms = _loss(new, torch.zeros(3), kinds, 2.0, torch.tensor([1.0, 0.0, 0.0]), option_weight=0.3, kl_beta=0.0)
    terms.loss.backward()
    # d(-J)/d log pi = -(A_seq + lambda * A_opt) on the decision, -A_seq on the other actions.
    assert torch.allclose(new.grad, torch.tensor([-2.3, -2.0, -2.0]))
    assert terms.n_decisions == 1 and terms.n_actions == 3


def test_loss_clips_the_probability_ratio():
    # Ratio far above 1 + eps with a positive advantage: the clipped branch is active, no gradient.
    new = torch.full((2,), 1.0, requires_grad=True)
    terms = _loss(new, torch.zeros(2), torch.tensor([GLUE, GLUE]), 1.0, torch.zeros(2), kl_beta=0.0)
    terms.loss.backward()
    assert torch.allclose(new.grad, torch.zeros(2)) and terms.clip_fraction == 1.0
    # With a negative advantage the unclipped branch is the minimum and the gradient pushes down.
    new = torch.full((2,), 1.0, requires_grad=True)
    terms = _loss(new, torch.zeros(2), torch.tensor([GLUE, GLUE]), -1.0, torch.zeros(2), kl_beta=0.0)
    terms.loss.backward()
    assert (new.grad > 0).all()


def test_kl_penalty_is_zero_at_the_reference_and_positive_elsewhere():
    new = torch.zeros(4, requires_grad=True)
    kinds = torch.full((4,), GLUE)
    at_ref = _loss(new, torch.zeros(4), kinds, 0.0, torch.zeros(4), logp_ref=[torch.zeros(4)], kl_beta=0.04)
    assert at_ref.kl == pytest.approx(0.0) and float(at_ref.loss.detach()) == pytest.approx(0.0)
    away = _loss(new, torch.zeros(4), kinds, 0.0, torch.zeros(4), logp_ref=[torch.full((4,), -1.0)], kl_beta=0.04)
    expected = 4 * (math.exp(-1.0) + 1.0 - 1.0)
    assert away.kl == pytest.approx(expected, rel=1e-5)
    assert float(away.loss.detach()) == pytest.approx(0.04 * expected / 4, rel=1e-5)


def test_micro_batches_accumulate_to_the_full_batch_gradient():
    torch.manual_seed(0)
    params = [torch.randn(5, requires_grad=True) for _ in range(4)]
    old = [torch.randn(5) for _ in range(4)]
    kinds = [torch.tensor([DECISION, GLUE, GLUE, DECISION, GLUE]) for _ in range(4)]
    seq_adv = torch.randn(4)
    opt_adv = [torch.randn(5) for _ in range(4)]
    ref = [torch.randn(5) for _ in range(4)]
    full = grpo_loss(params, old, kinds, seq_adv, opt_adv, ref)
    full.loss.backward()
    expected = [p.grad.clone() for p in params]
    for p in params:
        p.grad = None
    for chunk in ([0, 1], [2, 3]):
        part = grpo_loss(
            [params[i] for i in chunk], [old[i] for i in chunk], [kinds[i] for i in chunk], seq_adv[chunk],
            [opt_adv[i] for i in chunk], [ref[i] for i in chunk], n_sequences=4, n_actions=20,
        )
        part.loss.backward()
    for p, e in zip(params, expected):
        assert torch.allclose(p.grad, e, atol=1e-6)


def test_schedule_is_warmup_then_cosine():
    cfg = OptimConfig(warmup_steps=4)
    factors = [schedule_factor(step, 20, cfg) for step in range(20)]
    assert factors[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert all(a >= b for a, b in zip(factors[3:], factors[4:]))
    assert factors[-1] < 0.05
    assert schedule_factor(10, 20, OptimConfig(warmup_steps=4, schedule="constant")) == 1.0


def test_phases_for_the_method_and_its_ablations():
    full = build_phases(CurriculumConfig())
    assert [(p.level, p.steps, p.action_level, p.reward_level, p.certify) for p in full] == [
        (0, 5000, 0, 0, 0), (1, 10000, 1, 1, 1), (2, 15000, 2, 2, None),
    ]
    assert full[1].adaptive and not full[0].adaptive
    curriculum_only = build_phases(CurriculumConfig(use_options=False))
    assert [p.action_level for p in curriculum_only] == [0, 0, 0] and all(p.certify is None for p in curriculum_only)
    flat = build_phases(CurriculumConfig(use_options=False, use_curriculum=False))
    assert len(flat) == 1 and flat[0].steps == 30000 and flat[0].action_level == 0 and flat[0].level == -1
    options_only = build_phases(CurriculumConfig(use_curriculum=False))
    assert len(options_only) == 1 and options_only[0].action_level == 2


def test_length_pacer_advances_when_pass_rate_exceeds_threshold():
    pacer = LengthPacer(CurriculumConfig(advance_window=3, advance_threshold=0.5, max_composition_length=4))
    assert pacer.length == 2
    assert [pacer.update(r) for r in (0.9, 0.9)] == [False, False]  # window not full yet
    assert pacer.update(0.9) and pacer.length == 3
    assert not any(pacer.update(0.4) for _ in range(5)) and pacer.length == 3
    assert any(pacer.update(1.0) for _ in range(3)) and pacer.length == 4
    assert not any(pacer.update(1.0) for _ in range(5)) and pacer.length == 4  # capped at L_max


def test_pass_at_k_and_spearman():
    assert pass_at_k(10, 0, 1) == 0.0 and pass_at_k(10, 10, 5) == 1.0
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert pass_at_k(10, 3, 5) == pytest.approx(1 - 21 / 252)
    rho, p = spearman([1, 2, 3, 4, 5], [2, 4, 5, 4, 5])
    assert rho == pytest.approx(0.7378648, rel=1e-6) and p == pytest.approx(0.1546, abs=2e-3)
    rho, p = spearman([1, 2, 3, 4], [8, 6, 4, 2])
    assert rho == pytest.approx(-1.0) and p == 0.0
