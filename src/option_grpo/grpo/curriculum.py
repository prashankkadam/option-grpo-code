"""Curriculum phases and adaptive pacing.

The curriculum has three phases (Algorithm 4).  Phase ``k+1`` trains on the task
distribution ``D_k`` with action space ``A_k``; completing it certifies the options
``O_k`` that extend the action space of the next phase:

    C_0 --O_0--> C_1 --O_1--> C_2

Phase 2 (Level 1) additionally paces the composition length: it starts with paths of
``l_curr = 2`` calls and admits longer compositions whenever the recent pass rate
exceeds ``rho_adv``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from option_grpo.config import CurriculumConfig
from option_grpo.tasks.dataset import Task, TaskSet


@dataclass
class Phase:
    number: int  # 1-based phase number
    level: int  # curriculum level of the tasks (0, 1, 2); -1 mixes all levels
    steps: int
    action_level: int  # level of the action space A_k (0 = tokens only)
    reward_level: int  # which auxiliary reward terms are active
    adaptive: bool = False  # pace the composition length
    certify: int | None = None  # option level certified when the phase ends

    @property
    def name(self) -> str:
        return f"phase{self.number}-level{self.level if self.level >= 0 else 'all'}"


def build_phases(cfg: CurriculumConfig) -> list[Phase]:
    """Phases for the full method and for the ablations of the hierarchy.

    * options + curriculum (Option-GRPO): three phases, growing action space;
    * curriculum only: three phases, token-level actions throughout;
    * options only: one phase over all tasks with every candidate option available;
    * neither (flat GRPO): one token-level phase over all tasks.
    """
    n0, n1, n2 = cfg.steps
    if cfg.use_curriculum:
        if cfg.use_options:
            return [
                Phase(1, 0, n0, action_level=0, reward_level=0, certify=0),
                Phase(2, 1, n1, action_level=1, reward_level=1, adaptive=True, certify=1),
                Phase(3, 2, n2, action_level=2, reward_level=2),
            ]
        return [
            Phase(1, 0, n0, action_level=0, reward_level=0),
            Phase(2, 1, n1, action_level=0, reward_level=1, adaptive=True),
            Phase(3, 2, n2, action_level=0, reward_level=2),
        ]
    total = n0 + n1 + n2
    if cfg.use_options:
        return [Phase(1, -1, total, action_level=2, reward_level=2)]
    return [Phase(1, -1, total, action_level=0, reward_level=0)]


class LengthPacer:
    """Adaptive curriculum advancement inside Level 1 (Algorithm 4, lines 17-26)."""

    def __init__(self, cfg: CurriculumConfig) -> None:
        self.threshold = cfg.advance_threshold
        self.max_length = cfg.max_composition_length
        self.length = min(cfg.start_length, cfg.max_composition_length)
        self.window: deque[float] = deque(maxlen=max(cfg.advance_window, 1))

    def update(self, pass_rate: float) -> bool:
        """Record the pass rate of a step; returns True when the length was increased."""
        self.window.append(float(pass_rate))
        if self.length >= self.max_length or len(self.window) < self.window.maxlen:  # type: ignore[operator]
            return False
        if sum(self.window) / len(self.window) > self.threshold:
            self.length += 1
            self.window.clear()
            return True
        return False

    def state(self) -> dict[str, float | int | list[float]]:
        return {"length": self.length, "window": list(self.window)}

    def load_state(self, state: dict) -> None:
        self.length = int(state.get("length", self.length))
        self.window.clear()
        self.window.extend(state.get("window", []))


def phase_tasks(tasks: TaskSet, phase: Phase, split: str = "train", max_length: int | None = None) -> list[Task]:
    """Training tasks of a phase, optionally restricted to compositions of bounded length."""
    level = None if phase.level < 0 else phase.level
    selected = tasks.select(level=level, split=split, max_length=max_length)
    if not selected and max_length is not None:
        # No task is short enough yet: fall back to the shortest available compositions.
        pool = tasks.select(level=level, split=split)
        if pool:
            shortest = min(t.length for t in pool)
            selected = [t for t in pool if t.length == shortest]
    return selected
