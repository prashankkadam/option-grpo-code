"""Graph-based option discovery (Algorithm 1).

Level-0 candidates are one option per public SDK function.  Level-1 candidates are
mined from the parameter-flow graph: composable paths of two to ``L_max`` calls are
enumerated by depth-first search, weighted by the product of type-compatibility scores,
embedded as the mean identifier embedding of their functions, clustered agglomeratively
and represented by the highest-weight path of each cluster.
"""

from __future__ import annotations

import heapq

import numpy as np

from option_grpo.config import DiscoveryConfig
from option_grpo.graphs.build import SDKGraphs
from option_grpo.options.option import CANDIDATE, CodeOption, OptionVocabulary, make_option_key, option_io_types
from option_grpo.sdk.introspect import FunctionInfo
from option_grpo.sdk.types import type_compat
from option_grpo.utils import get_logger

log = get_logger(__name__)

Path = tuple[str, ...]


# --------------------------------------------------------------------------------------
# Helper (argument-constructor) names
# --------------------------------------------------------------------------------------
def helper_names(functions: list[FunctionInfo], graphs: SDKGraphs, min_compat: float = 0.5) -> frozenset[str]:
    """Call names that may appear inside the argument lists of the given path functions.

    These are the type constructors of the parameter types (``col`` for an ``Expr``
    parameter) and the methods that refine such a value (``Expr.alias``).
    """
    index = graphs.index
    path_ids = {fn.id for fn in functions}
    param_types = []
    for fn in functions:
        for param in fn.params:
            if param.type is None or param.type.is_primitive:
                continue
            param_types.extend(alt for alt in param.type.alternatives() if alt.class_refs())
    if not param_types:
        return frozenset()
    names: set[str] = set()
    for candidate in graphs.functions:
        if candidate.id in path_ids:
            continue
        produces = candidate.returns is not None and any(
            type_compat(candidate.returns, ptype, index.is_subclass) >= min_compat for ptype in param_types
        )
        if not produces:
            continue
        # Either a constructor of the parameter type, or a method that maps the type to itself.
        if candidate.receiver is None or any(
            type_compat(candidate.receiver, ptype, index.is_subclass) >= min_compat for ptype in param_types
        ):
            names.add(candidate.name)
    return frozenset(names)


# --------------------------------------------------------------------------------------
# Level 0
# --------------------------------------------------------------------------------------
def discover_level0(graphs: SDKGraphs, min_compat: float = 0.5) -> list[CodeOption]:
    """One atomic option per public SDK function."""
    options: list[CodeOption] = []
    for fn in graphs.functions:
        in_types, out_types, is_source = option_io_types(fn, fn)
        options.append(
            CodeOption(
                index=len(options),
                level=0,
                key=make_option_key(0, [fn]),
                path=(fn.id,),
                call_names=(fn.name,),
                in_types=in_types,
                out_types=out_types,
                is_source=is_source,
                helpers=helper_names([fn], graphs, min_compat),
                weight=1.0,
                members=[(fn.id,)],
                status=CANDIDATE,
            )
        )
    return options


# --------------------------------------------------------------------------------------
# Level 1
# --------------------------------------------------------------------------------------
def enumerate_paths(graphs: SDKGraphs, cfg: DiscoveryConfig) -> list[tuple[Path, float]]:
    """Composable simple paths of ``min_path_length``..``max_path_length`` functions.

    The weight of a path is ``w(p) = prod_{(f_a, f_b) in p} type_compat(f_a, f_b)``.
    The search expands the ``max_branch`` strongest successors of every node, prunes
    paths lighter than ``min_path_weight`` and keeps the ``max_paths_per_start`` heaviest
    paths per start node, which bounds the enumeration on densely connected SDKs.
    """
    successors = {fid: graphs.successors(fid)[: cfg.max_branch] for fid in graphs.flow.nodes}
    collected: list[tuple[Path, float]] = []
    for start in sorted(graphs.flow.nodes):
        best: list[tuple[float, int, Path]] = []  # min-heap on (weight, -length)
        stack: list[tuple[Path, float]] = [((start,), 1.0)]
        while stack:
            path, weight = stack.pop()
            if len(path) >= cfg.min_path_length:
                item = (weight, -len(path), path)
                if len(best) < cfg.max_paths_per_start:
                    heapq.heappush(best, item)
                elif item > best[0]:
                    heapq.heapreplace(best, item)
            if len(path) >= cfg.max_path_length:
                continue
            for nxt, edge_weight in successors[path[-1]]:
                if nxt in path:
                    continue
                new_weight = weight * edge_weight
                if new_weight < cfg.min_path_weight:
                    continue
                stack.append((path + (nxt,), new_weight))
        collected.extend((path, weight) for weight, _, path in best)
    collected.sort(key=lambda item: (-item[1], len(item[0]), item[0]))
    if len(collected) > cfg.max_paths:
        log.info("keeping the %d heaviest of %d enumerated paths", cfg.max_paths, len(collected))
        collected = collected[: cfg.max_paths]
    return collected


def cluster_paths(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """Agglomerative clustering (average linkage, cosine) cut at similarity ``threshold``.

    Two clusters are merged while their average pairwise cosine similarity is at least
    ``threshold``.  The dendrogram is built with the nearest-neighbour-chain algorithm,
    which is exact for average linkage and runs in ``O(n^2)`` time and memory.
    """
    n = embeddings.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    if n == 1:
        return np.zeros((1,), dtype=np.int64)
    unit = embeddings.astype(np.float32)
    norms = np.linalg.norm(unit, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = unit / norms
    with np.errstate(all="ignore"):  # some BLAS builds raise spurious FP flags in matmul
        sims = unit @ unit.T
    dist = np.clip(1.0 - sims, 0.0, 2.0).astype(np.float32)
    np.fill_diagonal(dist, np.inf)
    cut = max(1.0 - threshold, 0.0) + 1e-6

    size = np.ones(n, dtype=np.float32)
    active = np.ones(n, dtype=bool)
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    chain: list[int] = []
    for _ in range(n - 1):
        if not chain:
            chain.append(int(np.argmax(active)))
        while True:
            a = chain[-1]
            row = dist[a]
            b = int(np.argmin(row))
            if len(chain) >= 2 and row[chain[-2]] <= row[b]:
                b = chain[-2]  # prefer the previous element on ties so the chain terminates
            if len(chain) >= 2 and b == chain[-2]:
                break
            chain.append(b)
        a = chain.pop()
        b = chain.pop()
        height = float(dist[a, b])
        if height <= cut:
            parent[find(b)] = find(a)
        # Lance-Williams update for average linkage; ``b`` is absorbed into ``a``.
        merged = (size[a] * dist[a] + size[b] * dist[b]) / (size[a] + size[b])
        merged[~active] = np.inf
        merged[a] = np.inf
        merged[b] = np.inf
        dist[a, :] = merged
        dist[:, a] = merged
        dist[b, :] = np.inf
        dist[:, b] = np.inf
        size[a] += size[b]
        active[b] = False
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


def _merge_subsumed(options: list[CodeOption]) -> list[CodeOption]:
    """Enforce non-subsumption: no option's call set is a strict subset of another's."""
    ordered = sorted(options, key=lambda o: (-len(set(o.path)), -o.weight, o.key))
    kept: list[CodeOption] = []
    for option in ordered:
        calls = set(option.path)
        host = None
        for other in kept:
            if calls < set(other.path):
                if host is None or other.weight > host.weight:
                    host = other
        if host is None:
            kept.append(option)
        else:
            for member in option.members:
                if member not in host.members:
                    host.members.append(member)
            host.stats["merged"] = host.stats.get("merged", 0) + 1
    kept.sort(key=lambda o: (-o.weight, len(o.path), o.key))
    return kept


def discover_level1(graphs: SDKGraphs, cfg: DiscoveryConfig | None = None, min_compat: float = 0.5) -> list[CodeOption]:
    """Algorithm 1: Level-1 option candidates."""
    cfg = cfg or DiscoveryConfig()
    paths = enumerate_paths(graphs, cfg)
    if not paths:
        return []
    dim = len(next(iter(graphs.embeddings.values())))
    embeddings = np.zeros((len(paths), dim), dtype=np.float32)
    for row, (path, _) in enumerate(paths):
        embeddings[row] = np.mean([graphs.embeddings[fid] for fid in path], axis=0)  # e_p
    labels = cluster_paths(embeddings, cfg.cluster_threshold)
    clusters: dict[int, list[int]] = {}
    for row, label in enumerate(labels.tolist()):
        clusters.setdefault(label, []).append(row)

    index = graphs.index
    options: list[CodeOption] = []
    for rows in clusters.values():
        # Centroid p* = argmax_p w(p).  Exact type matches make ties common; they are
        # broken by proximity to the mean embedding of the cluster, then by length.
        top = max(paths[r][1] for r in rows)
        tied = [r for r in rows if paths[r][1] >= top - 1e-9]
        if len(tied) > 1:
            mean = embeddings[rows].mean(axis=0)
            mean /= max(float(np.linalg.norm(mean)), 1e-12)
            norms = np.linalg.norm(embeddings[tied], axis=1)
            norms[norms == 0.0] = 1.0
            closeness = (embeddings[tied] * mean).sum(axis=1) / norms
            order = sorted(
                range(len(tied)),
                key=lambda k: (-round(float(closeness[k]), 6), len(paths[tied[k]][0]), paths[tied[k]][0]),
            )
            centroid_row = tied[order[0]]
        else:
            centroid_row = tied[0]
        path, weight = paths[centroid_row]
        functions = [index.functions[fid] for fid in path]
        in_types, out_types, is_source = option_io_types(functions[0], functions[-1])
        options.append(
            CodeOption(
                index=-1,
                level=1,
                key=make_option_key(1, functions),
                path=path,
                call_names=tuple(fn.name for fn in functions),
                in_types=in_types,
                out_types=out_types,
                is_source=is_source,
                helpers=helper_names(functions, graphs, min_compat),
                weight=float(weight),
                members=[paths[r][0] for r in sorted(rows)[:64]],
                status=CANDIDATE,
                stats={"cluster_size": len(rows)},
            )
        )
    if cfg.merge_subsumed:
        options = _merge_subsumed(options)
    else:
        options.sort(key=lambda o: (-o.weight, len(o.path), o.key))
    log.info(
        "level-1 discovery for %s: %d paths -> %d clusters -> %d candidates",
        index.name, len(paths), len(clusters), len(options),
    )
    return options


def discover_options(graphs: SDKGraphs, cfg: DiscoveryConfig | None = None, min_compat: float = 0.5) -> OptionVocabulary:
    """Build the candidate vocabulary ``O_0^cand ∪ O_1^cand`` of an SDK."""
    vocab = OptionVocabulary(graphs.index.name)
    for option in discover_level0(graphs, min_compat):
        vocab.add(option)
    seen = {o.key for o in vocab.options}
    for option in discover_level1(graphs, cfg, min_compat):
        if option.key in seen:
            continue
        seen.add(option.key)
        vocab.add(option)
    vocab.build_composability(graphs.index, min_compat)
    return vocab
