"""Small shared helpers: logging, JSON/JSONL IO, seeding and identifier handling."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger; the root handler is installed once by :func:`setup_logging`."""
    return logging.getLogger(name)


def setup_logging(level: str | int = "INFO", log_file: str | os.PathLike[str] | None = None) -> None:
    """Configure root logging to stderr and, optionally, to a file."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(stream)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(fh)
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "filelock", "huggingface_hub", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (when available)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass


def stable_hash(*parts: Any, digits: int = 16) -> str:
    """Deterministic hex digest of the given parts (independent of ``PYTHONHASHSEED``)."""
    h = hashlib.sha256()
    for part in parts:
        h.update(repr(part).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:digits]


def stable_unit_float(*parts: Any) -> float:
    """Deterministic pseudo-random float in ``[0, 1)`` derived from ``parts``."""
    return int(stable_hash(*parts, digits=12), 16) / float(16**12)


def write_json(path: str | os.PathLike[str], obj: Any, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=False, ensure_ascii=False)
        f.write("\n")
    tmp.replace(p)


def read_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Any]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            n += 1
    tmp.replace(p)
    return n


def append_jsonl(path: str | os.PathLike[str], row: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def split_identifier(name: str) -> list[str]:
    """Split a ``snake_case`` / ``CamelCase`` / dotted identifier into lower-case words.

    >>> split_identifier("DataFrame.read_csv")
    ['data', 'frame', 'read', 'csv']
    """
    words: list[str] = []
    for chunk in re.split(r"[^A-Za-z0-9]+", name):
        if not chunk:
            continue
        for word in _CAMEL_BOUNDARY.split(chunk):
            if word:
                words.append(word.lower())
    return words


def first_sentence(doc: str | None) -> str:
    """First sentence of a docstring, whitespace-normalised, without the trailing period."""
    if not doc:
        return ""
    text = doc.strip().split("\n\n", 1)[0]
    text = " ".join(text.split())
    match = re.search(r"(?<=[^A-Z.])\.(\s|$)", text)
    if match:
        text = text[: match.start()]
    return text.strip().rstrip(".")


def chunked(items: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
