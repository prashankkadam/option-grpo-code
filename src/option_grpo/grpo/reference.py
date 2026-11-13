"""The reference policy ``pi_ref`` of the KL penalty."""

from __future__ import annotations

import copy
from contextlib import contextmanager

import torch

from option_grpo.policy.hierarchical import HierarchicalPolicy
from option_grpo.policy.sampling import RolloutEngine, Trajectory
from option_grpo.utils import get_logger

log = get_logger(__name__)


class ReferencePolicy:
    """Frozen reference used to regularise the policy.

    * ``copy`` -- a frozen snapshot of the whole policy (language model and option head),
      refreshed at the start of each curriculum phase.  Needs memory for a second copy.
    * ``adapter_off`` -- for LoRA training: the base model with adapters disabled plus a
      snapshot of the option head; no second copy of the language model is needed.
    * ``none`` -- no reference (the KL term is dropped).
    """

    def __init__(self, policy: HierarchicalPolicy, mode: str = "copy") -> None:
        self.policy = policy
        self.mode = mode
        self.frozen: HierarchicalPolicy | None = None
        self.option_snapshot: dict[str, torch.Tensor] | None = None

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def refresh(self) -> None:
        """Take a new snapshot of the current policy."""
        if self.mode == "copy":
            self.frozen = None  # release the previous snapshot first
            frozen = copy.deepcopy(self.policy)
            frozen.eval()
            for param in frozen.parameters():
                param.requires_grad_(False)
            self.frozen = frozen
        elif self.mode == "adapter_off":
            self.option_snapshot = self.policy.option_state()

    @contextmanager
    def _adapters_disabled(self):
        policy = self.policy
        current = policy.option_state()
        assert self.option_snapshot is not None
        policy.load_option_state(self.option_snapshot)
        was_training = policy.training
        policy.eval()
        try:
            with policy.lm.disable_adapter():
                yield
        finally:
            policy.load_option_state(current)
            policy.train(was_training)

    @torch.no_grad()
    def log_probs(self, engine: RolloutEngine, trajectories: list[Trajectory]) -> list[torch.Tensor] | None:
        """``log pi_ref(a_t | s_t)`` for the recorded actions, or None when disabled."""
        if self.mode == "none":
            return None
        if self.mode == "copy":
            if self.frozen is None:
                self.refresh()
            return engine.log_probs(trajectories, self.frozen)
        if self.option_snapshot is None:
            self.refresh()
        with self._adapters_disabled():
            return [lp.clone() for lp in engine.log_probs(trajectories, self.policy)]
