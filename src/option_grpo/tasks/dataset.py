"""Curriculum tasks and task sets."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from option_grpo.sdk.types import TypeRef
from option_grpo.utils import read_jsonl, write_jsonl


@dataclass
class Task:
    """One problem of UnseenSDK-Bench.

    ``path`` is the SDK-graph path the task was generated from.  ``reference`` is the
    call skeleton synthesised from that path; it is used to validate the task and to
    measure the reference length ``T_ref`` and is never shown to the policy.
    """

    id: str
    sdk: str
    level: int  # 0 atomic call, 1 composed pattern, 2 full program
    path: list[str]  # function ids
    call_names: list[str]
    description: str  # q
    prelude: str  # import lines
    setup: str  # c_setup
    fixtures: list[dict[str, Any]] = field(default_factory=list)
    setup_types: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)  # runtime type acceptance spec
    tests: list[dict[str, Any]] = field(default_factory=list)  # property tests (Algorithm 3)
    variants: list[dict[str, Any]] = field(default_factory=list)  # edge-case setups
    reference: str | None = None
    n_min: int = 0  # minimal number of SDK calls
    t_ref: int | None = None  # reference length in tokens
    split: str = "train"
    validated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Composition length: number of calls on the generating path."""
        return len(self.path)

    def setup_value_types(self) -> list[TypeRef]:
        out = []
        for data in self.setup_types.values():
            ref = TypeRef.from_json(data)
            if ref is not None:
                out.append(ref)
        return out

    def context_code(self) -> str:
        """Code that precedes the solution (imports and setup), for static analysis."""
        return f"{self.prelude}\n{self.setup}\n"

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Task:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


class TaskSet:
    """The tasks of one SDK across the three curriculum levels."""

    def __init__(self, tasks: Iterable[Task] = ()) -> None:
        self.tasks: list[Task] = list(tasks)
        self._by_id = {t.id: t for t in self.tasks}

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks)

    def get(self, task_id: str) -> Task:
        return self._by_id[task_id]

    def select(
        self,
        level: int | None = None,
        split: str | None = None,
        max_length: int | None = None,
    ) -> list[Task]:
        out = []
        for task in self.tasks:
            if level is not None and task.level != level:
                continue
            if split not in (None, "all") and task.split != split:
                continue
            if max_length is not None and task.length > max_length:
                continue
            out.append(task)
        return out

    def counts(self) -> dict[str, int]:
        table: dict[str, int] = {}
        for task in self.tasks:
            key = f"L{task.level}/{task.split}"
            table[key] = table.get(key, 0) + 1
        return dict(sorted(table.items()))

    def save(self, path: str | Path) -> None:
        write_jsonl(path, (t.to_json() for t in self.tasks))

    @classmethod
    def load(cls, path: str | Path) -> TaskSet:
        return cls(Task.from_json(row) for row in read_jsonl(path))


class TaskSampler:
    """Uniform sampling without replacement within an epoch, reshuffled when exhausted."""

    def __init__(self, tasks: list[Task], seed: int = 0) -> None:
        if not tasks:
            raise ValueError("cannot sample from an empty task list")
        self.tasks = list(tasks)
        self.rng = random.Random(seed)
        self._queue: list[Task] = []

    def set_tasks(self, tasks: list[Task]) -> None:
        if not tasks:
            raise ValueError("cannot sample from an empty task list")
        self.tasks = list(tasks)
        self._queue = []

    def sample(self, n: int = 1) -> list[Task]:
        out: list[Task] = []
        while len(out) < n:
            if not self._queue:
                self._queue = list(self.tasks)
                self.rng.shuffle(self._queue)
            out.append(self._queue.pop())
        return out
