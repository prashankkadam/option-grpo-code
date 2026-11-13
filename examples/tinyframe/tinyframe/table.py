"""Materialised tables."""

from __future__ import annotations

from typing import Any


class Table:
    """An in-memory, materialised table of rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = [dict(row) for row in rows]

    def __len__(self) -> int:
        return len(self._rows)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Table) and self._rows == other._rows

    def __repr__(self) -> str:
        return f"Table(rows={len(self._rows)}, columns={self.column_names()})"

    @property
    def num_rows(self) -> int:
        return len(self._rows)

    def column_names(self) -> list[str]:
        """Names of the columns, in first-seen order."""
        names: list[str] = []
        for row in self._rows:
            for key in row:
                if key not in names:
                    names.append(key)
        return names

    def column(self, name: str) -> list[Any]:
        """Values of one column."""
        return [row.get(name) for row in self._rows]

    def to_pylist(self) -> list[dict[str, Any]]:
        """Rows as a list of dictionaries."""
        return [dict(row) for row in self._rows]

    def to_pydict(self) -> dict[str, list[Any]]:
        """Columns as a dictionary of lists."""
        return {name: self.column(name) for name in self.column_names()}
