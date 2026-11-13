"""Lazy frames."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tinyframe.expr import Expr
from tinyframe.table import Table

Rows = list[dict[str, Any]]


class Frame:
    """A lazy sequence of row transformations; nothing runs until :meth:`collect`."""

    def __init__(self, source: Callable[[], Rows]) -> None:
        self._source = source

    def _derive(self, fn: Callable[[Rows], Rows]) -> Frame:
        source = self._source
        return Frame(lambda: fn(source()))

    # ------------------------------------------------------------------ row-wise
    def where(self, predicate: Expr) -> Frame:
        """Filter rows, keeping those for which the predicate holds."""
        return self._derive(lambda rows: [row for row in rows if predicate.evaluate(row)])

    def select(self, *columns: str) -> Frame:
        """Project the frame onto the given columns."""
        assert len(columns) > 0, "select needs at least one column"
        return self._derive(lambda rows: [{c: row.get(c) for c in columns} for row in rows])

    def with_column(self, name: str, expr: Expr) -> Frame:
        """Add a computed column."""
        return self._derive(lambda rows: [{**row, name: expr.evaluate(row)} for row in rows])

    def sort(self, by: str, desc: bool = False) -> Frame:
        """Sort rows by a column."""
        return self._derive(
            lambda rows: sorted(rows, key=lambda row: (row.get(by) is None, row.get(by)), reverse=desc)
        )

    def limit(self, n: int) -> Frame:
        """Keep the first ``n`` rows."""
        assert n >= 0, "limit must be non-negative"
        return self._derive(lambda rows: rows[:n])

    def distinct(self) -> Frame:
        """Drop duplicate rows."""

        def unique(rows: Rows) -> Rows:
            seen: set[tuple[tuple[str, Any], ...]] = set()
            out: Rows = []
            for row in rows:
                key = tuple(sorted((k, repr(v)) for k, v in row.items()))
                if key not in seen:
                    seen.add(key)
                    out.append(row)
            return out

        return self._derive(unique)

    # ------------------------------------------------------------------ relational
    def groupby(self, key: str) -> GroupedFrame:
        """Group rows by the values of a column."""
        return GroupedFrame(self, key)

    def join(self, other: Frame, on: str) -> Frame:
        """Inner-join with another frame on a shared column."""

        def joined(rows: Rows) -> Rows:
            right = other._source()
            out: Rows = []
            for left_row in rows:
                for right_row in right:
                    if left_row.get(on) == right_row.get(on):
                        out.append({**right_row, **left_row})
            return out

        return self._derive(joined)

    # ------------------------------------------------------------------ actions
    def collect(self) -> Table:
        """Execute the plan and materialise the result."""
        return Table(self._source())

    def count_rows(self) -> int:
        """Execute the plan and count the resulting rows."""
        return len(self._source())


class GroupedFrame:
    """Rows grouped by a key column, waiting for an aggregation."""

    def __init__(self, frame: Frame, key: str) -> None:
        self._frame = frame
        self._key = key

    def agg(self, expr: Expr) -> Frame:
        """Aggregate every group with an aggregation expression such as ``col("x").mean()``."""
        assert expr.aggregate is not None, "agg needs an aggregation expression"
        key = self._key

        def aggregate(rows: Rows) -> Rows:
            groups: dict[Any, list[Any]] = {}
            for row in rows:
                groups.setdefault(row.get(key), []).append(expr.evaluate(row))
            out: Rows = []
            for group, values in groups.items():
                present = [v for v in values if v is not None]
                if expr.aggregate == "count":
                    value: Any = len(present)
                elif not present:
                    value = None
                elif expr.aggregate == "mean":
                    value = sum(present) / len(present)
                elif expr.aggregate == "sum":
                    value = sum(present)
                elif expr.aggregate == "max":
                    value = max(present)
                else:
                    value = min(present)
                out.append({key: group, expr.name: value})
            return out

        return self._frame._derive(aggregate)

    def count(self) -> Frame:
        """Number of rows in every group."""
        key = self._key

        def counted(rows: Rows) -> Rows:
            counts: dict[Any, int] = {}
            for row in rows:
                counts[row.get(key)] = counts.get(row.get(key), 0) + 1
            return [{key: group, "count": n} for group, n in counts.items()]

        return self._frame._derive(counted)
