"""A tiny embedded table store."""

from __future__ import annotations

from typing import Any

from tinyframe._io import from_records
from tinyframe.frame import Frame
from tinyframe.table import Table

_DATABASES: dict[str, dict[str, Table]] = {}


class Query:
    """A pending lookup against a stored table."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def limit(self, n: int) -> Query:
        """Keep at most ``n`` matches."""
        assert n >= 0, "limit must be non-negative"
        return Query(self._rows[:n])

    def to_list(self) -> list[dict[str, Any]]:
        """Run the query and return the matching rows."""
        return [dict(row) for row in self._rows]


class StoredTable:
    """A named table living inside a :class:`Database`."""

    def __init__(self, name: str, table: Table) -> None:
        self._name = name
        self._table = table

    @property
    def name(self) -> str:
        return self._name

    def search(self, column: str, value: Any) -> Query:
        """Find the rows whose ``column`` equals ``value``."""
        return Query([row for row in self._table.to_pylist() if row.get(column) == value])

    def to_frame(self) -> Frame:
        """Expose the stored rows as a lazy frame."""
        return from_records(self._table.to_pylist())

    def count_rows(self) -> int:
        """Number of stored rows."""
        return len(self._table)


class Database:
    """A collection of named tables addressed by a URI."""

    def __init__(self, uri: str) -> None:
        self._uri = uri
        self._tables = _DATABASES.setdefault(uri, {})

    def create_table(self, name: str, data: Table) -> StoredTable:
        """Create (or replace) a table from materialised data."""
        self._tables[name] = data
        return StoredTable(name, data)

    def open_table(self, name: str) -> StoredTable:
        """Open an existing table."""
        if name not in self._tables:
            raise KeyError(f"no table named {name!r}")
        return StoredTable(name, self._tables[name])

    def table_names(self) -> list[str]:
        """Names of all tables in the database."""
        return sorted(self._tables)


def connect(uri: str) -> Database:
    """Open the database identified by ``uri``."""
    return Database(uri)
