"""Curriculum-staged Option-GRPO."""

from option_grpo.grpo.advantages import option_advantages, sequence_advantages
from option_grpo.grpo.curriculum import LengthPacer, Phase, build_phases, phase_tasks
from option_grpo.grpo.loss import LossTerms, grpo_loss
from option_grpo.grpo.optim import apply_partial_freeze, build_optimizer, build_param_groups, schedule_factor, set_learning_rates
from option_grpo.grpo.reference import ReferencePolicy
from option_grpo.grpo.trainer import Group, OptionGRPOTrainer, TrainState, relative_gradient_variance

__all__ = [
    "Group",
    "LengthPacer",
    "LossTerms",
    "OptionGRPOTrainer",
    "Phase",
    "ReferencePolicy",
    "TrainState",
    "apply_partial_freeze",
    "build_optimizer",
    "build_param_groups",
    "build_phases",
    "grpo_loss",
    "option_advantages",
    "phase_tasks",
    "relative_gradient_variance",
    "schedule_factor",
    "sequence_advantages",
    "set_learning_rates",
]
