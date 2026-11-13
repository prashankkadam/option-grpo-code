"""Curriculum tasks: input synthesis, task generation, prompts and task sets."""

from option_grpo.tasks.dataset import Task, TaskSampler, TaskSet
from option_grpo.tasks.generator import TaskGenerator
from option_grpo.tasks.prompts import BEGIN_MARKER, END_MARKER, build_prompt, extract_solution
from option_grpo.tasks.synth import SynthError, Synthesizer

__all__ = [
    "BEGIN_MARKER",
    "END_MARKER",
    "SynthError",
    "Synthesizer",
    "Task",
    "TaskGenerator",
    "TaskSampler",
    "TaskSet",
    "build_prompt",
    "extract_solution",
]
