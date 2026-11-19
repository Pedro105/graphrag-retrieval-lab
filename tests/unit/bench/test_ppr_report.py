import numpy as np
from scipy import sparse

from graphrag_lab.bench.graph import personalized_pagerank
from graphrag_lab.bench.report import markdown_tables, summarize


def path_graph(n: int) -> sparse.csr_matrix:
    rows = list(range(n - 1)) + list(range(1, n))
    cols = list(range(1, n)) + list(range(n - 1))
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def test_ppr_is_a_distribution_that_decays_with_distance_from_the_seed() -> None:
    # Seed in the middle of a 7-node path, so neighbours have equal degree.
    scores = personalized_pagerank(path_graph(7), seeds=[3])
    assert abs(scores.sum() - 1.0) < 1e-6
    assert scores[3] > scores[2] > scores[1] > scores[0]
    assert np.allclose(scores[:3], scores[4:][::-1])  # symmetric around the seed


def test_ppr_without_seeds_is_all_zero() -> None:
    assert not personalized_pagerank(path_graph(3), seeds=[]).any()


def test_ppr_ignores_components_unreachable_from_seeds() -> None:
    graph = sparse.block_diag([path_graph(3), path_graph(3)]).tocsr()
    scores = personalized_pagerank(graph, seeds=[0])
    assert scores[3:].sum() == 0.0


def record(retriever: str, budget: int, hit: bool | None, recall: float) -> dict[str, object]:
    return {
        "retriever": retriever,
        "unit_kind": "triple",
        "budget": budget,
        "tokens": 10,
        "answer_recall": hit,
        "support_recall": recall,
        "full_support": recall == 1.0,
        "seed_hit": True,
    }


def test_summarize_skips_undefined_answer_recall_and_renders_tables() -> None:
    records = [
        record("graph_ppr", 64, True, 1.0),
        record("graph_ppr", 64, None, 0.5),
        record("graph_ppr", 128, False, 0.0),
        record("graph_ppr", 128, None, 1.0),
    ]
    summary = summarize(records, resamples=50)
    by_budget = {row["budget"]: row for row in summary}

    assert by_budget[64]["answer_recall"]["n"] == 1
    assert by_budget[64]["answer_recall"]["mean"] == 1.0
    assert by_budget[64]["support_recall"]["mean"] == 0.75
    assert by_budget[128]["full_support"]["mean"] == 0.5
    assert by_budget[64]["seed_hit"] == 1.0

    tables = markdown_tables(summary, ["graph_ppr", "not_run"])
    assert "| `graph_ppr` | triple | 100.0" in tables
    assert "not_run" not in tables
    assert "Seed hit" in tables
