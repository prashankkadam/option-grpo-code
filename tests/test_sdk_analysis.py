"""Static SDK analysis, type compatibility and the three graphs."""

from __future__ import annotations

import numpy as np

from option_grpo.config import DiscoveryConfig
from option_grpo.graphs.build import flow_compat
from option_grpo.options.discovery import cluster_paths, discover_level1, enumerate_paths
from option_grpo.sdk.types import ANY, TypeRef, parse_annotation, type_compat


def test_parse_annotation_normalises_typing_forms():
    assert str(parse_annotation("Optional[List[int]]")) == "list[int] | None"
    assert str(parse_annotation("Union[str, 'Path', None]")) == "str | Path | None"
    assert str(parse_annotation("dict[str, list[Any]]")) == "dict[str, list[Any]]"
    assert parse_annotation("Annotated[int, 'meta']") == TypeRef("int")
    assert parse_annotation(None) is None
    assert parse_annotation("this is not ( valid") == ANY
    assert parse_annotation("int | None").alternatives()[1].is_none


def test_type_compat_scores():
    frame = TypeRef("Frame", qualname="pkg.Frame", sdk=True)
    lazy = TypeRef("LazyFrame", qualname="pkg.LazyFrame", sdk=True)
    other = TypeRef("Table", qualname="pkg.Table", sdk=True)
    is_sub = lambda child, parent: (child, parent) == ("pkg.LazyFrame", "pkg.Frame")  # noqa: E731
    assert type_compat(frame, frame) == 1.0
    assert type_compat(lazy, frame, is_sub) == 0.9
    assert type_compat(frame, lazy, is_sub) == 0.3
    assert type_compat(frame, other, is_sub) == 0.0
    # Optional parameters accept the value at full score, multi-class unions slightly less.
    assert type_compat(frame, parse_annotation("Frame | None")) == 1.0
    assert type_compat(frame, TypeRef("Union", (frame, other))) == 0.8
    # Primitive flows carry no structural signal and unknown types score low.
    assert type_compat(TypeRef("str"), TypeRef("str")) == 0.0
    assert type_compat(frame, ANY) == 0.1
    assert type_compat(TypeRef("None"), frame) == 0.0
    # Containers of SDK objects flow into compatible containers.
    assert type_compat(TypeRef("list", (frame,)), TypeRef("Sequence", (frame,))) == 1.0
    assert type_compat(TypeRef("list", (TypeRef("int"),)), TypeRef("list", (TypeRef("int"),))) == 0.0


def test_index_contains_public_api_with_resolved_types(index):
    fn = index.functions["tinyframe.frame.Frame.where"]
    assert fn.public_path == "tinyframe.Frame.where"
    assert fn.kind == "method" and fn.receiver.qualname == "tinyframe.frame.Frame"
    assert fn.params[0].type.qualname == "tinyframe.expr.Expr" and fn.params[0].type.sdk
    assert fn.returns.qualname == "tinyframe.frame.Frame"
    # Functions defined in a private module are public through the package re-export.
    assert index.functions["tinyframe._io.read_csv"].public_path == "tinyframe.read_csv"
    # Constructors are callables named after their class.
    ctor = index.constructor("tinyframe.table.Table")
    assert ctor.name == "Table" and ctor.returns.qualname == "tinyframe.table.Table"
    # Assertions over parameters are harvested for invariant tests.
    assert index.functions["tinyframe.frame.Frame.limit"].asserts == ["n >= 0"]
    names = {f.name for f in index.public_functions()}
    assert {"read_csv", "where", "collect", "col", "connect"} <= names
    assert not any(name.startswith("_") for name in names)
    assert "num_rows" in index.namespace_names() and "Frame" in index.namespace_names()


def test_flow_graph_edges_follow_types(index, graphs):
    read_csv = index.functions["tinyframe._io.read_csv"]
    where = index.functions["tinyframe.frame.Frame.where"]
    col = index.functions["tinyframe.expr.col"]
    collect = index.functions["tinyframe.frame.Frame.collect"]
    assert flow_compat(read_csv, where, index) == (1.0, ["self"])
    assert flow_compat(col, where, index) == (1.0, ["predicate"])
    assert flow_compat(collect, where, index)[0] == 0.0
    assert graphs.flow.has_edge(read_csv.id, where.id)
    assert not graphs.flow.has_edge(collect.id, where.id)
    assert 0.0 < graphs.density() < 1.0
    producers = [fn.name for fn, _ in graphs.producers(where.params[0].type)]
    assert "col" in producers and "lit" in producers


def test_cluster_paths_matches_bruteforce_average_linkage():
    rng = np.random.default_rng(0)

    def brute(points, threshold):
        unit = points / np.linalg.norm(points, axis=1, keepdims=True)
        dist = 1.0 - unit @ unit.T
        clusters = [[i] for i in range(len(points))]
        while len(clusters) > 1:
            best = None
            for a in range(len(clusters)):
                for b in range(a + 1, len(clusters)):
                    d = float(np.mean([dist[i, j] for i in clusters[a] for j in clusters[b]]))
                    if best is None or d < best[0]:
                        best = (d, a, b)
            if best[0] > 1.0 - threshold + 1e-6:
                break
            clusters[best[1]] += clusters.pop(best[2])
        labels = np.zeros(len(points), dtype=int)
        for label, members in enumerate(clusters):
            labels[members] = label
        return labels

    for _ in range(10):
        centres = rng.normal(size=(4, 8))
        points = np.concatenate([c + 0.25 * rng.normal(size=(5, 8)) for c in centres])
        for threshold in (0.5, 0.8, 0.9):
            ours = cluster_paths(points.astype(np.float32), threshold)
            ref = brute(points, threshold)
            same = (ours[:, None] == ours[None, :]) == (ref[:, None] == ref[None, :])
            assert same.all()


def test_level1_discovery_respects_algorithm_constraints(graphs):
    cfg = DiscoveryConfig()
    paths = enumerate_paths(graphs, cfg)
    assert paths and all(2 <= len(p) <= cfg.max_path_length for p, _ in paths)
    assert all(len(set(p)) == len(p) for p, _ in paths)  # simple paths
    for path, weight in paths[:200]:
        product = 1.0
        for a, b in zip(path, path[1:]):
            product *= graphs.flow[a][b]["weight"]
        assert abs(product - weight) < 1e-9
    options = discover_level1(graphs, cfg)
    assert options
    sets = [set(o.path) for o in options]
    # Non-subsumption: no option's call set is a strict subset of another's.
    assert not any(a < b for a in sets for b in sets)
    labels = {o.label for o in options}
    assert any("read_csv" in label and "collect" in label for label in labels)


def test_candidate_vocabulary_and_composability(vocab, index):
    assert len(vocab.level(0)) == len(index.public_functions())
    assert vocab.action_space(1) == [] and vocab.action_space(2) == []
    read_csv = vocab.get("L0:tinyframe.read_csv")
    where = vocab.get("L0:tinyframe.Frame.where")
    collect = vocab.get("L0:tinyframe.Frame.collect")
    assert read_csv.is_source and not where.is_source
    assert "col" in where.helpers and "where" not in where.helpers
    assert where.index in vocab.successors[read_csv.index]
    assert where.index not in vocab.successors[collect.index]
    restored = type(vocab).from_json(vocab.to_json())
    assert [o.key for o in restored.options] == [o.key for o in vocab.options]
    assert restored.options[where.index].in_types == where.in_types
