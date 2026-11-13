"""Column expressions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Row = dict[str, Any]


class Expr:
    """A lazily evaluated expression over the columns of a row."""

    def __init__(self, fn: Callable[[Row], Any], name: str, aggregate: str | None = None) -> None:
        self._fn = fn
        self._name = name
        self._aggregate = aggregate

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, row: Row) -> Any:
        """Evaluate the expression on a single row."""
        return self._fn(row)

    @property
    def name(self) -> str:
        return self._name

    @property
    def aggregate(self) -> str | None:
        return self._aggregate

    # ------------------------------------------------------------------ naming
    def alias(self, name: str) -> Expr:
        """Rename the output column of the expression."""
        assert name, "alias must be a non-empty string"
        return Expr(self._fn, name, self._aggregate)

    # ------------------------------------------------------------------ comparisons
    def gt(self, value: float) -> Expr:
        """Keep values strictly greater than ``value``."""
        return Expr(lambda row: _cmp(self._fn(row), value, lambda a, b: a > b), self._name)

    def lt(self, value: float) -> Expr:
        """Keep values strictly less than ``value``."""
        return Expr(lambda row: _cmp(self._fn(row), value, lambda a, b: a < b), self._name)

    def eq(self, value: Any) -> Expr:
        """Keep values equal to ``value``."""
        return Expr(lambda row: self._fn(row) == value, self._name)

    def between(self, low: float, high: float) -> Expr:
        """Keep values in the closed interval ``[low, high]``."""
        assert low <= high, "low must not exceed high"
        return Expr(lambda row: _cmp(self._fn(row), low, lambda a, b: a >= b) and _cmp(self._fn(row), high, lambda a, b: a <= b), self._name)

    def is_null(self) -> Expr:
        """True where the value is missing."""
        return Expr(lambda row: self._fn(row) is None, self._name)

    def __gt__(self, other: Any) -> Expr:
        return self.gt(other)

    def __lt__(self, other: Any) -> Expr:
        return self.lt(other)

    # ------------------------------------------------------------------ aggregations
    def mean(self) -> Expr:
        """Average of the values within each group."""
        return Expr(self._fn, self._name, "mean")

    def sum(self) -> Expr:
        """Sum of the values within each group."""
        return Expr(self._fn, self._name, "sum")

    def max(self) -> Expr:
        """Largest value within each group."""
        return Expr(self._fn, self._name, "max")

    def min(self) -> Expr:
        """Smallest value within each group."""
        return Expr(self._fn, self._name, "min")

    def count(self) -> Expr:
        """Number of non-missing values within each group."""
        return Expr(self._fn, self._name, "count")


def _cmp(left: Any, right: Any, op: Callable[[Any, Any], bool]) -> bool:
    if left is None or right is None:
        return False
    return bool(op(left, right))


def col(name: str) -> Expr:
    """Reference a column by name."""
    assert isinstance(name, str) and name, "column name must be a non-empty string"
    return Expr(lambda row: row.get(name), name)


def lit(value: Any) -> Expr:
    """Wrap a constant as an expression."""
    return Expr(lambda row: value, "literal")
