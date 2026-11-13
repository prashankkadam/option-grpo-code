"""Optimiser, learning-rate schedule and the partial freeze between phases.

Parameters are split once into fixed groups (option head, token embeddings, one group
per transformer block, everything else).  The schedule and the partial freeze then act
by rescaling the learning rate of those groups, so the optimiser is never rebuilt:

* the schedule is linear warm-up followed by cosine decay, restarted at the beginning
  of each curriculum phase (``schedule_scope="phase"``) or run once (``"global"``);
* after phase ``k`` the bottom ``bottom_fraction[k]`` of the blocks (and the token
  embeddings) either continue at the slow rate ``option_lr`` (``mode="slow"``) or stop
  being updated (``mode="hard"``).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from option_grpo.config import FreezeConfig, OptimConfig
from option_grpo.policy.hierarchical import HierarchicalPolicy
from option_grpo.utils import get_logger

log = get_logger(__name__)


def build_param_groups(policy: HierarchicalPolicy, cfg: OptimConfig) -> list[dict[str, Any]]:
    """Fixed parameter groups; every group carries a ``role`` and an ``lr_scale``."""
    assigned: set[int] = set()
    groups: list[dict[str, Any]] = []

    def add(role: str, params: list[torch.nn.Parameter], decay: bool, layer: int | None = None) -> None:
        unique: dict[int, torch.nn.Parameter] = {}
        for param in params:
            if param.requires_grad and id(param) not in assigned:
                unique.setdefault(id(param), param)
        params = list(unique.values())
        if not params:
            return
        assigned.update(unique)
        groups.append(
            {
                "params": params,
                "lr": cfg.lr,
                "base_lr": cfg.lr,
                "lr_scale": 1.0,
                "weight_decay": cfg.weight_decay if decay else 0.0,
                "role": role,
                "layer": layer,
            }
        )

    def split(role: str, params: list[torch.nn.Parameter], layer: int | None = None) -> None:
        add(role, [p for p in params if p.ndim >= 2], True, layer)
        add(role, [p for p in params if p.ndim < 2], False, layer)

    add("option", policy.option_parameters(), False)
    # Token embeddings (and the output head when its weight is tied to them).
    add("embedding", list(policy.token_embeddings.parameters()), False)
    for number, layer in enumerate(policy.transformer_layers()):
        split("layer", list(layer.parameters()), number)
    split("other", list(policy.parameters()))
    return groups


def build_optimizer(policy: HierarchicalPolicy, cfg: OptimConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(build_param_groups(policy, cfg), lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps)


def schedule_factor(step: int, total: int, cfg: OptimConfig) -> float:
    """Multiplier of the base learning rate at ``step`` (0-based) of a ``total``-step run."""
    warmup = min(cfg.warmup_steps, max(total - 1, 0))
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if cfg.schedule == "constant" or total <= warmup:
        return 1.0
    progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine


def set_learning_rates(optimizer: torch.optim.Optimizer, factor: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = group.get("base_lr", group["lr"]) * group.get("lr_scale", 1.0) * factor


def apply_partial_freeze(
    optimizer: torch.optim.Optimizer,
    policy: HierarchicalPolicy,
    completed_phases: int,
    freeze: FreezeConfig,
    optim: OptimConfig,
) -> int:
    """Rescale the learning rates of the groups frozen after ``completed_phases`` phases.

    Returns the number of transformer blocks in the slow / frozen set.
    """
    if freeze.mode == "none" or completed_phases <= 0 or not freeze.bottom_fraction:
        return 0
    fraction = freeze.bottom_fraction[min(completed_phases, len(freeze.bottom_fraction)) - 1]
    n_layers = len(policy.transformer_layers())
    cutoff = int(math.floor(fraction * n_layers + 1e-9))
    scale = 0.0 if freeze.mode == "hard" else optim.option_lr / optim.lr
    for group in optimizer.param_groups:
        role = group.get("role")
        slow = (role == "layer" and group.get("layer") is not None and group["layer"] < cutoff) or (
            role == "embedding" and freeze.freeze_token_embeddings and cutoff > 0
        )
        if slow:
            group["lr_scale"] = scale
            if freeze.mode == "hard":
                for param in group["params"]:
                    param.requires_grad_(False)
    log.info(
        "partial freeze after phase %d: bottom %d of %d blocks -> %s",
        completed_phases, cutoff, n_layers, "frozen" if freeze.mode == "hard" else f"lr x{scale:g}",
    )
    return cutoff
