"""The Option-GRPO objective.

For a group of ``G`` sampled trajectories the surrogate that is maximised is

    J = 1/G * sum_i [ sum_t  clip(A_seq_i, rho_{i,t})
                      + lambda * sum_{j in decisions(i)} clip(A_opt(o_j), rho_{i,j}) ]
        - beta * KL(pi_theta || pi_ref)

with the PPO clipped term ``clip(A, rho) = min(rho * A, clamp(rho, 1-eps, 1+eps) * A)``
and the probability ratio ``rho = pi_theta(a|s) / pi_old(a|s)``.  The first sum runs
over every action of the trajectory (option decisions, intra-option tokens and glue
tokens); the second adds per-option credit on the option-decision positions only.  The
KL term uses the estimator ``pi_ref/pi_theta - log(pi_ref/pi_theta) - 1`` on the
sampled actions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from option_grpo.policy.sampling import DECISION


@dataclass
class LossTerms:
    loss: torch.Tensor  # scalar to minimise (already normalised)
    surrogate_seq: float
    surrogate_opt: float
    kl: float
    clip_fraction: float
    approx_kl_old: float
    n_actions: int
    n_decisions: int


def grpo_loss(
    logp_new: list[torch.Tensor],
    logp_old: list[torch.Tensor],
    kinds: list[torch.Tensor],
    seq_adv: torch.Tensor,
    opt_adv: list[torch.Tensor],
    logp_ref: list[torch.Tensor] | None = None,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
    option_weight: float = 0.3,
    length_normalize: bool = False,
    n_sequences: int | None = None,
    n_actions: int | None = None,
) -> LossTerms:
    """Loss of a (micro-)batch of trajectories.

    ``n_sequences`` and ``n_actions`` are the totals of the *whole* batch when it is
    processed in several micro-batches, so that the accumulated gradient equals the
    gradient of the full-batch objective.
    """
    total_sequences = n_sequences if n_sequences is not None else len(logp_new)
    total_actions = n_actions if n_actions is not None else sum(int(lp.numel()) for lp in logp_new)
    device = seq_adv.device
    objective = torch.zeros((), device=device)
    kl_sum = torch.zeros((), device=device)
    stat_seq = stat_opt = 0.0
    clipped = 0
    approx_kl = 0.0
    count = 0
    decisions = 0
    for i, new in enumerate(logp_new):
        if new.numel() == 0:
            continue
        old = logp_old[i].to(new)
        ratio = torch.exp(new - old)
        clamped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        adv = seq_adv[i].to(new)
        seq_term = torch.minimum(ratio * adv, clamped * adv).sum()
        if length_normalize:
            seq_term = seq_term / new.numel()
        is_decision = kinds[i].to(device) == DECISION
        option_adv = opt_adv[i].to(new)
        opt_term = (torch.minimum(ratio * option_adv, clamped * option_adv) * is_decision).sum()
        objective = objective + seq_term + option_weight * opt_term
        if logp_ref is not None and kl_beta > 0:
            log_ratio = logp_ref[i].to(new) - new
            kl_sum = kl_sum + (torch.exp(log_ratio) - log_ratio - 1.0).sum()
        with torch.no_grad():
            stat_seq += float(seq_term)
            stat_opt += float(opt_term)
            clipped += int(((ratio - 1.0).abs() > clip_eps).sum())
            approx_kl += float((old - new).sum())
            count += int(new.numel())
            decisions += int(is_decision.sum())
    loss = -objective / max(total_sequences, 1)
    if logp_ref is not None and kl_beta > 0:
        loss = loss + kl_beta * kl_sum / max(total_actions, 1)
    return LossTerms(
        loss=loss,
        surrogate_seq=stat_seq,
        surrogate_opt=stat_opt,
        kl=float(kl_sum.detach()),
        clip_fraction=clipped / max(count, 1),
        approx_kl_old=approx_kl / max(count, 1),
        n_actions=count,
        n_decisions=decisions,
    )
