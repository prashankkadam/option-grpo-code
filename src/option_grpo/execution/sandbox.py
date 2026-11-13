"""Sandboxed execution of generated programs.

Two backends run the stand-alone harness (:mod:`option_grpo.execution.harness`):

* :class:`SubprocessSandbox` -- a fresh Python process per program in a temporary
  working directory, with a wall-clock timeout, an address-space limit (Linux) and
  sockets disabled inside the process.  Suitable for development and trusted SDKs.
* :class:`DockerSandbox` -- one container per program with ``--network none``, a
  memory limit and a read-only root file system.  This is the configuration used for
  the benchmark (30 s timeout, 2 GB memory, no network).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from option_grpo.config import ExecutionConfig
from option_grpo.sdk.introspect import SDKIndex
from option_grpo.tasks.dataset import Task
from option_grpo.utils import get_logger

log = get_logger(__name__)

HARNESS_PATH = Path(__file__).with_name("harness.py")

# Stages in increasing order of success; see the harness for their meaning.
STAGE_ORDER = (
    "parse_error",
    "import_error",
    "setup_error",
    "timeout",
    "crash",
    "runtime_error",
    "output_none",
    "type_mismatch",
    "type_ok",
    "tests_passed",
)


@dataclass
class ExecResult:
    """Outcome of executing one program on one task."""

    stage: str
    error: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    tests: list[dict[str, Any]] = field(default_factory=list)
    n_api: int = 0
    duration_s: float = 0.0

    @property
    def passed(self) -> bool:
        return self.stage == "tests_passed"

    @property
    def executed(self) -> bool:
        return self.stage in {"output_none", "type_mismatch", "type_ok", "tests_passed"}

    @property
    def output_digest(self) -> str | None:
        return self.output.get("digest") if self.output else None

    @property
    def tests_passed(self) -> int:
        return sum(1 for t in self.tests if t.get("passed"))

    def to_json(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "error": self.error,
            "output": self.output,
            "trace": self.trace,
            "tests": self.tests,
            "n_api": self.n_api,
            "duration_s": self.duration_s,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ExecResult:
        return cls(
            stage=data.get("stage") or "crash",
            error=data.get("error"),
            output=data.get("output"),
            trace=list(data.get("trace") or []),
            tests=list(data.get("tests") or []),
            n_api=int(data.get("n_api") or 0),
            duration_s=float(data.get("duration_s") or 0.0),
        )


def build_job(
    task: Task,
    solution: str,
    index: SDKIndex,
    cfg: ExecutionConfig,
    sys_path: list[str] | None = None,
    setup: str | None = None,
    tests: list[dict[str, Any]] | None = None,
    fixtures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Job description consumed by the harness."""
    return {
        "task_id": task.id,
        "sdk_roots": [index.import_name],
        "sys_path": list(sys_path or []),
        "prelude": task.prelude,
        "setup": task.setup if setup is None else setup,
        "fixtures": task.fixtures if fixtures is None else fixtures,
        "solution": solution,
        "expected": task.expected or {"any": True},
        "tests": task.tests if tests is None else tests,
        "variants": task.variants,
        "functions": task.meta.get("functions", []),
        "limits": {"memory_mb": cfg.memory_mb, "allow_network": cfg.allow_network},
    }


class Sandbox(ABC):
    """Executes harness jobs; implementations must be thread-safe."""

    def __init__(self, cfg: ExecutionConfig) -> None:
        self.cfg = cfg
        self._pool = ThreadPoolExecutor(max_workers=max(cfg.workers, 1), thread_name_prefix="sandbox")

    @abstractmethod
    def run(self, job: dict[str, Any]) -> ExecResult:
        """Execute one job."""

    def run_many(self, jobs: list[dict[str, Any]]) -> list[ExecResult]:
        """Execute jobs concurrently, preserving order."""
        if not jobs:
            return []
        return list(self._pool.map(self.run, jobs))

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ shared helpers
    @staticmethod
    def _prepare(job: dict[str, Any]) -> Path:
        workdir = Path(tempfile.mkdtemp(prefix="ogrpo-"))
        shutil.copyfile(HARNESS_PATH, workdir / "_harness.py")
        with (workdir / "_job.json").open("w", encoding="utf-8") as handle:
            json.dump(job, handle)
        return workdir

    @staticmethod
    def _collect(workdir: Path, returncode: int | None, stderr: str) -> ExecResult:
        result_path = workdir / "_result.json"
        if result_path.exists():
            try:
                with result_path.open("r", encoding="utf-8") as handle:
                    return ExecResult.from_json(json.load(handle))
            except (OSError, json.JSONDecodeError):
                pass
        return ExecResult(
            stage="crash",
            error={"type": "HarnessCrash", "message": f"exit code {returncode}: {stderr[-500:]}", "line": None},
        )


class SubprocessSandbox(Sandbox):
    """One Python subprocess per program."""

    def __init__(self, cfg: ExecutionConfig) -> None:
        super().__init__(cfg)
        self.python = cfg.python or sys.executable

    def run(self, job: dict[str, Any]) -> ExecResult:
        workdir = self._prepare(job)
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(workdir),
            "TMPDIR": str(workdir),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        for key in ("LANG", "LC_ALL", "VIRTUAL_ENV", "CONDA_PREFIX", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            if key in os.environ:
                env[key] = os.environ[key]
        try:
            process = subprocess.Popen(
                [self.python, "-B", "_harness.py", "_job.json", "_result.json"],
                cwd=workdir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
                text=True,
            )
            try:
                _, stderr = process.communicate(timeout=self.cfg.timeout_s)
            except subprocess.TimeoutExpired:
                self._kill(process)
                process.communicate()
                return ExecResult(
                    stage="timeout",
                    error={"type": "Timeout", "message": f"exceeded {self.cfg.timeout_s:.0f}s", "line": None},
                    duration_s=self.cfg.timeout_s,
                )
            return self._collect(workdir, process.returncode, stderr or "")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @staticmethod
    def _kill(process: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                process.kill()
            except ProcessLookupError:
                pass


class DockerSandbox(Sandbox):
    """One container per program: no network, bounded memory, read-only root."""

    def __init__(self, cfg: ExecutionConfig) -> None:
        super().__init__(cfg)
        if not cfg.docker_image:
            raise ValueError("execution.docker_image must be set for the docker backend")
        if shutil.which("docker") is None:
            raise RuntimeError("the docker executable was not found on PATH")
        self.image = cfg.docker_image

    def run(self, job: dict[str, Any]) -> ExecResult:
        workdir = self._prepare(job)
        os.chmod(workdir, 0o777)
        name = f"ogrpo-{uuid.uuid4().hex[:12]}"
        command = [
            "docker", "run", "--rm", "--name", name,
            "--memory", f"{self.cfg.memory_mb}m", "--memory-swap", f"{self.cfg.memory_mb}m",
            "--cpus", "1", "--pids-limit", "256", "--read-only",
            "--tmpfs", "/tmp:rw,size=256m",
            "--security-opt", "no-new-privileges",
            "-e", "HOME=/work", "-e", "TMPDIR=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONHASHSEED=0",
            "-v", f"{workdir}:/work", "-w", "/work",
        ]
        if not self.cfg.allow_network:
            command += ["--network", "none"]
        command += list(self.cfg.docker_extra_args)
        command += [self.image, "python", "-B", "_harness.py", "_job.json", "_result.json"]
        try:
            try:
                completed = subprocess.run(
                    command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    text=True, timeout=self.cfg.timeout_s + 10.0,
                )
            except subprocess.TimeoutExpired:
                subprocess.run(["docker", "kill", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                return ExecResult(
                    stage="timeout",
                    error={"type": "Timeout", "message": f"exceeded {self.cfg.timeout_s:.0f}s", "line": None},
                    duration_s=self.cfg.timeout_s,
                )
            result = self._collect(workdir, completed.returncode, completed.stderr or "")
            if result.stage == "crash" and completed.returncode == 137:
                result.error = {"type": "OutOfMemory", "message": "container killed (memory limit)", "line": None}
            return result
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def build_sandbox(cfg: ExecutionConfig) -> Sandbox:
    if cfg.backend == "docker":
        return DockerSandbox(cfg)
    return SubprocessSandbox(cfg)


class Executor:
    """Runs candidate solutions of tasks through a sandbox, with result caching."""

    def __init__(
        self,
        index: SDKIndex,
        cfg: ExecutionConfig,
        sandbox: Sandbox | None = None,
        sys_path: list[str] | None = None,
        cache_size: int = 20_000,
    ) -> None:
        self.index = index
        self.cfg = cfg
        self.sandbox = sandbox or build_sandbox(cfg)
        self.sys_path = list(sys_path or [])
        self._cache: dict[tuple[str, str], ExecResult] = {}
        self._cache_size = cache_size

    def run(self, items: list[tuple[Task, str]]) -> list[ExecResult]:
        """Execute ``(task, solution)`` pairs; identical pairs are executed once."""
        results: list[ExecResult | None] = [None] * len(items)
        pending: dict[tuple[str, str], list[int]] = {}
        for position, (task, solution) in enumerate(items):
            key = (task.id, solution)
            cached = self._cache.get(key)
            if cached is not None:
                results[position] = cached
            else:
                pending.setdefault(key, []).append(position)
        if pending:
            keys = list(pending)
            jobs = []
            for key in keys:
                task, solution = items[pending[key][0]]
                jobs.append(build_job(task, solution, self.index, self.cfg, self.sys_path))
            for key, result in zip(keys, self.sandbox.run_many(jobs)):
                if result.stage not in {"timeout", "crash"}:
                    if len(self._cache) >= self._cache_size:
                        self._cache.pop(next(iter(self._cache)))
                    self._cache[key] = result
                for position in pending[key]:
                    results[position] = result
        return results  # type: ignore[return-value]

    def close(self) -> None:
        self.sandbox.close()
