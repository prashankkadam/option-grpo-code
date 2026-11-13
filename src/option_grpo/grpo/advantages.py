"""Group-relative advantages.

Sequence level (standard GRPO): ``A_i = (R_i - mean(R)) / (std(R) + eps)`` over the
``G`` samples of a prompt.

Option level: ``A_opt(o) = mean_{i in S_o^+} R_i - mean_{i in S_o^-} R_i`` where
``S_o^+`` are the samples of the group whose option sequence contains ``o`` and
``S_o^-`` the remaining ones.  An option used by every sample, or by none, carries no
group-relative information and receives a zero advantage.
"""

from __future__ import annotations

import torch


def sequence_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Group-normalised advantages for a 1-D tensor of rewards."""
    rewards = rewards.float()
    if rewards.numel() < 2:
        return torch.zeros_like(rewards)
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + eps)


def option_advantages(
    rewards: list[float],
    option_sets: list[set[int]],
    normalize: bool = False,
    eps: float = 1e-4,
) -> dict[int, float]:
    """Advantage of every option that occurs in the group."""
    group = len(rewards)
    if group == 0:
        return {}
    total = float(sum(rewards))
    scale = 1.0
    if normalize:
        scale = float(torch.tensor(rewards).float().std(unbiased=False)) + eps
    out: dict[int, float] = {}
    for option in set().union(*option_sets) if option_sets else set():
        inside = [r for r, used in zip(rewards, option_sets) if option in used]
        n_in = len(inside)
        n_out = group - n_in
        if n_in == 0 or n_out == 0:
            out[option] = 0.0
            continue
        sum_in = float(sum(inside))
        out[option] = (sum_in / n_in - (total - sum_in) / n_out) / scale
    return out
