"""Construction of the three SDK graphs (Stage 0 of Algorithm 4).

* ``G_api``  -- bipartite type graph: which functions consume and produce which types.
* ``G_flow`` -- parameter-flow graph: ``f_a -> f_b`` when the value returned by ``f_a``
  can be passed to ``f_b``; the edge weight is ``type_compat(f_a, f_b)``.
* ``G_sem``  -- semantic similarity graph over function identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from option_grpo.config import GraphConfig
from option_grpo.graphs.embed import Embedder, build_embedder, identifier_text
from option_grpo.sdk.introspect import FunctionInfo, SDKIndex
from option_grpo.sdk.types import DEFAULT_WEIGHTS, CompatWeights, TypeRef, type_compat
from option_grpo.utils import get_logger, read_json, write_json

log = get_logger(__name__)


def flow_compat(
    src: FunctionInfo,
    dst: FunctionInfo,
    index: SDKIndex,
    weights: CompatWeights = DEFAULT_WEIGHTS,
) -> tuple[float, list[str]]:
    """``type_compat(f_a, f_b)``: best score of feeding ``src``'s result into ``dst``.

    Returns the score and the names of the parameters of ``dst`` that achieve it
    (``"self"`` denotes the receiver of a method).
    """
    if src.returns is None or src.returns.is_none:
        return 0.0, []
    best = 0.0
    via: list[str] = []
    for name, ptype in dst.input_types():
        if ptype is None:
            continue
        score = type_compat(src.returns, ptype, index.is_subclass, weights)
        if score > best + 1e-9:
            best, via = score, [name]
        elif score > 0 and abs(score - best) <= 1e-9:
            via.append(name)
    return best, via


@dataclass
class SDKGraphs:
    """The graphs of one SDK together with the index they were built from."""

    index: SDKIndex
    api: nx.DiGraph
    flow: nx.DiGraph
    sem: nx.Graph
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ statistics
    @property
    def functions(self) -> list[FunctionInfo]:
        return [self.index.functions[fid] for fid in self.flow.nodes]

    def density(self) -> float:
        """Composability density ``|E_compose| / |F|^2``."""
        n = self.flow.number_of_nodes()
        return self.flow.number_of_edges() / float(n * n) if n else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "sdk": self.index.name,
            "version": self.index.version,
            "functions": self.flow.number_of_nodes(),
            "flow_edges": self.flow.number_of_edges(),
            "semantic_edges": self.sem.number_of_edges(),
            "types": sum(1 for _, d in self.api.nodes(data=True) if d.get("kind") == "type"),
            "density": round(self.density(), 4),
        }

    # ------------------------------------------------------------------ queries
    def successors(self, fid: str) -> list[tuple[str, float]]:
        """Functions that can consume the result of ``fid``, by decreasing weight."""
        out = [(dst, data["weight"]) for _, dst, data in self.flow.out_edges(fid, data=True)]
        out.sort(key=lambda item: (-item[1], item[0]))
        return out

    def producers(self, target: TypeRef, min_compat: float = 0.5) -> list[tuple[FunctionInfo, float]]:
        """Type constructors from ``G_api``: public callables whose result fits ``target``."""
        out: list[tuple[FunctionInfo, float]] = []
        for fn in self.functions:
            if fn.returns is None:
                continue
            score = type_compat(fn.returns, target, self.index.is_subclass)
            if score >= min_compat:
                out.append((fn, score))
        out.sort(key=lambda item: (-item[1], len(item[0].required_params), item[0].id))
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.index.save(directory / "index.json")
        write_json(
            directory / "graphs.json",
            {
                "meta": self.meta,
                "stats": self.stats(),
                "nodes": list(self.flow.nodes),
                "flow": [
                    {"src": u, "dst": v, "weight": d["weight"], "via": d.get("via", [])}
                    for u, v, d in self.flow.edges(data=True)
                ],
                "sem": [{"a": u, "b": v, "sim": d["weight"]} for u, v, d in self.sem.edges(data=True)],
            },
            indent=1,
        )
        ids = list(self.embeddings)
        matrix = np.stack([self.embeddings[i] for i in ids]) if ids else np.zeros((0, 1), dtype=np.float32)
        np.savez_compressed(directory / "embeddings.npz", ids=np.array(ids, dtype=object), vectors=matrix)

    @classmethod
    def load(cls, directory: str | Path) -> SDKGraphs:
        directory = Path(directory)
        index = SDKIndex.load(directory / "index.json")
        data = read_json(directory / "graphs.json")
        flow = nx.DiGraph()
        flow.add_nodes_from(data["nodes"])
        for edge in data["flow"]:
            flow.add_edge(edge["src"], edge["dst"], weight=edge["weight"], via=edge.get("via", []))
        sem = nx.Graph()
        sem.add_nodes_from(data["nodes"])
        for edge in data["sem"]:
            sem.add_edge(edge["a"], edge["b"], weight=edge["sim"])
        embeddings: dict[str, np.ndarray] = {}
        emb_path = directory / "embeddings.npz"
        if emb_path.exists():
            with np.load(emb_path, allow_pickle=True) as blob:
                for fid, vector in zip(blob["ids"].tolist(), blob["vectors"]):
                    embeddings[str(fid)] = np.asarray(vector, dtype=np.float32)
        functions = [index.functions[fid] for fid in data["nodes"]]
        return cls(index, build_api_graph(index, functions), flow, sem, embeddings, data.get("meta", {}))


def build_api_graph(index: SDKIndex, functions: list[FunctionInfo]) -> nx.DiGraph:
    """Bipartite graph between types and the functions that consume / produce them."""
    graph = nx.DiGraph()

    def type_node(ref: TypeRef) -> str:
        node = f"type:{ref.key}"
        if node not in graph:
            graph.add_node(node, kind="type", name=ref.name, qualname=ref.qualname, sdk=ref.sdk)
        return node

    for cls in index.classes.values():
        if cls.public_path is None:
            continue
        ref = TypeRef(cls.name, qualname=cls.qualname, sdk=True)
        node = type_node(ref)
        for base in cls.bases:
            if base in index.classes:
                base_cls = index.classes[base]
                graph.add_edge(node, type_node(TypeRef(base_cls.name, qualname=base, sdk=True)), kind="inherits")
    for fn in functions:
        fnode = f"func:{fn.id}"
        graph.add_node(fnode, kind="function", id=fn.id, name=fn.name)
        for pname, ptype in fn.input_types():
            if ptype is None:
                continue
            for ref in ptype.class_refs():
                graph.add_edge(type_node(ref), fnode, kind="accepts", param=pname)
        if fn.returns is not None:
            for ref in fn.returns.class_refs():
                graph.add_edge(fnode, type_node(ref), kind="returns")
    return graph


def build_flow_graph(
    index: SDKIndex,
    functions: list[FunctionInfo],
    min_compat: float,
    weights: CompatWeights = DEFAULT_WEIGHTS,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(fn.id for fn in functions)
    for src in functions:
        if src.returns is None or src.returns.is_none or src.returns.is_unknown:
            continue
        for dst in functions:
            score, via = flow_compat(src, dst, index, weights)
            if score >= min_compat:
                graph.add_edge(src.id, dst.id, weight=float(score), via=via)
    return graph


def build_semantic_graph(
    functions: list[FunctionInfo],
    embedder: Embedder,
    threshold: float,
) -> tuple[nx.Graph, dict[str, np.ndarray]]:
    graph = nx.Graph()
    graph.add_nodes_from(fn.id for fn in functions)
    if not functions:
        return graph, {}
    vectors = embedder.encode([identifier_text(fn) for fn in functions])
    with np.errstate(all="ignore"):  # some BLAS builds raise spurious FP flags in matmul
        sims = vectors @ vectors.T
    rows, cols = np.where(np.triu(sims, k=1) >= threshold)
    for i, j in zip(rows.tolist(), cols.tolist()):
        graph.add_edge(functions[i].id, functions[j].id, weight=float(sims[i, j]))
    return graph, {fn.id: vectors[k] for k, fn in enumerate(functions)}


def build_graphs(
    index: SDKIndex,
    cfg: GraphConfig | None = None,
    embedder: Embedder | None = None,
    weights: CompatWeights = DEFAULT_WEIGHTS,
) -> SDKGraphs:
    """Construct ``G_api``, ``G_flow`` and ``G_sem`` for an indexed SDK."""
    cfg = cfg or GraphConfig()
    functions = index.public_functions(include_properties=cfg.include_properties)
    if not functions:
        raise RuntimeError(f"SDK {index.name!r} exposes no public callables")
    embedder = embedder or build_embedder(cfg.embedder)
    flow = build_flow_graph(index, functions, cfg.min_compat, weights)
    sem, embeddings = build_semantic_graph(functions, embedder, cfg.semantic_threshold)
    graphs = SDKGraphs(
        index=index,
        api=build_api_graph(index, functions),
        flow=flow,
        sem=sem,
        embeddings=embeddings,
        meta={
            "min_compat": cfg.min_compat,
            "semantic_threshold": cfg.semantic_threshold,
            "embedder": type(embedder).__name__ if cfg.embedder in {"hashing", "hash"} else cfg.embedder,
        },
    )
    log.info("graphs for %s: %s", index.name, graphs.stats())
    return graphs
