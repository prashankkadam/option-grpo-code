"""Readers."""

from __future__ import annotations

import csv
import json
from typing import Any

from tinyframe.frame import Frame


def _convert(value: str) -> Any:
    if value == "":
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def read_csv(path: str, delimiter: str = ",") -> Frame:
    """Read a CSV file into a lazy frame."""

    def load() -> list[dict[str, Any]]:
        with open(path, newline="", encoding="utf-8") as handle:
            return [{k: _convert(v) for k, v in row.items()} for row in csv.DictReader(handle, delimiter=delimiter)]

    return Frame(load)


def read_json(path: str) -> Frame:
    """Read a JSON file holding a list of records into a lazy frame."""

    def load() -> list[dict[str, Any]]:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("expected a JSON array of records")
        return [dict(row) for row in data]

    return Frame(load)


def from_records(records: list[dict[str, Any]]) -> Frame:
    """Build a lazy frame from a list of dictionaries."""
    snapshot = [dict(row) for row in records]
    return Frame(lambda: [dict(row) for row in snapshot])
