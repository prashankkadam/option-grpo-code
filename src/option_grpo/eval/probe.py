"""Memorisation probe (contamination analysis).

For each SDK, identifier-completion prompts of the form

    In <SDK>, the <noun> that <verb phrase> is called

are built for classes, methods and free functions drawn uniformly from the public API.
The frozen base model completes each prompt greedily and the completion is correct when
its first identifier equals the canonical name (case-insensitive, ignoring leading dots
and module prefixes).  Top-1 recall far below that of well-known control libraries
indicates that the model has no usable knowledge of the SDK.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Protocol

from option_grpo.sdk.introspect import SDKIndex
from option_grpo.utils import first_sentence, stable_hash

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_NO_ES = ("ss", "sh", "ch", "x", "z", "o")


@dataclass
class Probe:
    prompt: str
    answer: str
    kind: str  # class | method | function
    target: str  # public path of the probed identifier


def third_person(phrase: str) -> str:
    """``"Read a CSV file"`` -> ``"reads a CSV file"`` (first word conjugated)."""
    words = phrase.strip().split()
    if not words:
        return ""
    verb = words[0].lower()
    if verb.endswith("s") and not verb.endswith("ss"):
        conjugated = verb  # already third person ("returns", "creates")
    elif verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
        conjugated = verb[:-1] + "ies"
    elif verb.endswith(_NO_ES):
        conjugated = verb + "es"
    else:
        conjugated = verb + "s"
    return " ".join([conjugated, *words[1:]])


def build_probes(index: SDKIndex, sdk_title: str, n: int = 200, seed: int = 0) -> list[Probe]:
    """``n`` probes split evenly across classes, methods and free functions."""
    pools: dict[str, list[tuple[str, str, str]]] = {"class": [], "method": [], "function": []}
    for cls in index.classes.values():
        phrase = first_sentence(cls.doc)
        if cls.public_path and phrase:
            pools["class"].append((cls.name, f"represents {phrase[0].lower()}{phrase[1:]}", cls.public_path))
    for fn in index.public_functions():
        phrase = first_sentence(fn.doc)
        if not phrase or fn.kind == "constructor":
            continue
        kind = "method" if fn.owner else "function"
        pools[kind].append((fn.name, third_person(phrase), fn.public_path or fn.id))
    rng = random.Random(stable_hash("probe", index.name, seed))
    probes: list[Probe] = []
    share = n // 3
    quotas = {"class": share, "method": share, "function": n - 2 * share}
    leftovers = 0
    for kind, pool in pools.items():
        rng.shuffle(pool)
        take = min(quotas[kind], len(pool))
        leftovers += quotas[kind] - take
        for name, phrase, target in pool[:take]:
            probes.append(Probe(f"In {sdk_title}, the {kind} that {phrase} is called", name, kind, target))
        pools[kind] = pool[take:]
    for kind, pool in pools.items():  # redistribute unused quota
        while leftovers > 0 and pool:
            name, phrase, target = pool.pop()
            probes.append(Probe(f"In {sdk_title}, the {kind} that {phrase} is called", name, kind, target))
            leftovers -= 1
    return probes


def is_correct(completion: str, answer: str) -> bool:
    """Does the first identifier of the completion name the probed API?"""
    match = _IDENTIFIER.search(completion.strip().lstrip("`'\"*. \n"))
    if match is None:
        return False
    return match.group(0).split(".")[-1].lower() == answer.lower()


class Completer(Protocol):
    def complete(self, prompt: str, max_new_tokens: int) -> str: ...


class PolicyCompleter:
    """Greedy completion with a :class:`~option_grpo.policy.sampling.RolloutEngine`."""

    def __init__(self, engine) -> None:
        self.engine = engine

    def complete(self, prompt: str, max_new_tokens: int = 12) -> str:
        (trajectory,) = self.engine.rollout("probe", prompt, 1, None, None, temperature=0.0, max_new_tokens=max_new_tokens)
        return trajectory.text


def run_probe(probes: list[Probe], completer: Completer, max_new_tokens: int = 12) -> dict[str, Any]:
    """Top-1 identifier recall overall and per identifier kind."""
    hits: dict[str, list[bool]] = {}
    rows = []
    for probe in probes:
        completion = completer.complete(probe.prompt, max_new_tokens)
        correct = is_correct(completion, probe.answer)
        hits.setdefault(probe.kind, []).append(correct)
        rows.append({"prompt": probe.prompt, "answer": probe.answer, "completion": completion, "correct": correct})
    total = [h for values in hits.values() for h in values]
    return {
        "n": len(total),
        "recall": sum(total) / len(total) if total else 0.0,
        "by_kind": {kind: sum(values) / len(values) for kind, values in hits.items() if values},
        "rows": rows,
    }
