"""Small statistics helpers: the unbiased pass@k estimator and Spearman correlation."""

from __future__ import annotations

import math
from collections.abc import Sequence


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of pass@k from ``c`` correct samples out of ``n``.

    ``pass@k = 1 - C(n - c, k) / C(n, k)``: the probability that at least one of ``k``
    samples drawn without replacement from the ``n`` generated ones is correct.
    """
    if k > n:
        raise ValueError(f"pass@{k} needs at least {k} samples, got {n}")
    if n - c < k:
        return 1.0
    value = 1.0
    for i in range(n - c + 1, n + 1):
        value *= 1.0 - k / i
    return 1.0 - value


def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks (1-based), ties sharing the mean of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0
        for position in range(i, j + 1):
            ranks[order[position]] = mean_rank
        i = j + 1
    return ranks


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_test_p_value(r: float, n: int) -> float:
    """Two-sided p-value of a correlation ``r`` over ``n`` points (Student t, n-2 dof)."""
    if n < 3:
        return math.nan
    if abs(r) >= 1.0:
        return 0.0
    dof = n - 2
    t = r * math.sqrt(dof / (1.0 - r * r))
    return _betainc(dof / 2.0, 0.5, dof / (dof + t * t))


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return math.nan
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0.0 or syy <= 0.0:
        return math.nan
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Spearman rank correlation and its two-sided p-value."""
    if len(x) != len(y):
        raise ValueError("spearman needs sequences of equal length")
    rho = pearson(_ranks(x), _ranks(y))
    if math.isnan(rho):
        return math.nan, math.nan
    return rho, t_test_p_value(rho, len(x))
