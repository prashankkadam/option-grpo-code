"""Option description length (Definition 3) and option coverage.

``ODL_H(c)`` is the minimum number of options whose sequential execution is
observationally equivalent to ``c``.  It is computed on the API-call abstraction of the
program: the chain-level SDK calls of ``c`` must be segmented into the call paths of
options from ``O_0 ∪ O_1``, and the minimum number of segments is found by dynamic
programming.  ``ODL`` is infinite when no segmentation exists, when more than ``m_max``
options would be needed, or when the program relies on third-party libraries or
re-implements SDK functionality by hand (such a program is not expressible as a
sequence of options at all).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from option_grpo.options.option import CodeOption


@dataclass
class Decomposition:
    odl: float  # math.inf when no decomposition exists
    segments: list[tuple[int, int, CodeOption]]  # (start, end, option) over the call sequence

    @property
    def finite(self) -> bool:
        return math.isfinite(self.odl)

    @property
    def options(self) -> list[CodeOption]:
        return [segment[2] for segment in self.segments]


def decompose(
    sequence: list[str],
    patterns: dict[tuple[str, ...], CodeOption],
    max_options: int = 8,
) -> Decomposition:
    """Minimum-cardinality segmentation of ``sequence`` into option call paths."""
    n = len(sequence)
    if n == 0:
        return Decomposition(0.0, [])
    by_length: dict[int, dict[tuple[str, ...], CodeOption]] = {}
    for names, option in patterns.items():
        by_length.setdefault(len(names), {})[names] = option
    best: list[float] = [math.inf] * (n + 1)
    back: list[tuple[int, CodeOption] | None] = [None] * (n + 1)
    best[0] = 0.0
    for end in range(1, n + 1):
        for length, table in by_length.items():
            start = end - length
            if start < 0 or not math.isfinite(best[start]):
                continue
            option = table.get(tuple(sequence[start:end]))
            if option is None:
                continue
            cost = best[start] + 1.0
            # Ties favour the longer (higher-level) option.
            if cost < best[end] or (cost == best[end] and back[end] is not None and length > end - back[end][0]):
                best[end] = cost
                back[end] = (start, option)
    if not math.isfinite(best[n]) or best[n] > max_options:
        return Decomposition(math.inf, [])
    segments: list[tuple[int, int, CodeOption]] = []
    cursor = n
    while cursor > 0:
        start, option = back[cursor]  # type: ignore[misc]
        segments.append((start, cursor, option))
        cursor = start
    segments.reverse()
    return Decomposition(best[n], segments)


def option_description_length(
    sequence: list[str],
    patterns: dict[tuple[str, ...], CodeOption],
    max_options: int = 8,
    expressible: bool = True,
) -> float:
    """``ODL_H(c)`` for a program with chain-level call sequence ``sequence``.

    ``expressible`` is False for programs that use third-party libraries or hand-written
    re-implementations; their ODL is infinite by definition.
    """
    if not expressible:
        return math.inf
    return decompose(sequence, patterns, max_options).odl


def cover(
    sequence: list[str],
    patterns: dict[tuple[str, ...], CodeOption],
    min_len: int = 1,
) -> list[tuple[int, int, CodeOption]]:
    """Maximum coverage of ``sequence`` by non-overlapping option matches.

    Unlike :func:`decompose`, calls that match no option may be skipped.  Among covers
    with the same number of covered calls the one with fewer options is returned.
    """
    n = len(sequence)
    by_length: dict[int, dict[tuple[str, ...], CodeOption]] = {}
    for names, option in patterns.items():
        if len(names) >= min_len:
            by_length.setdefault(len(names), {})[names] = option
    # score = (covered calls, -number of options)
    best: list[tuple[int, int]] = [(0, 0)] * (n + 1)
    back: list[tuple[int, CodeOption | None]] = [(0, None)] * (n + 1)
    for end in range(1, n + 1):
        best[end] = best[end - 1]
        back[end] = (end - 1, None)
        for length, table in by_length.items():
            start = end - length
            if start < 0:
                continue
            option = table.get(tuple(sequence[start:end]))
            if option is None:
                continue
            score = (best[start][0] + length, best[start][1] - 1)
            if score > best[end]:
                best[end] = score
                back[end] = (start, option)
    matches: list[tuple[int, int, CodeOption]] = []
    cursor = n
    while cursor > 0:
        start, option = back[cursor]
        if option is not None:
            matches.append((start, cursor, option))
        cursor = start
    matches.reverse()
    return matches
