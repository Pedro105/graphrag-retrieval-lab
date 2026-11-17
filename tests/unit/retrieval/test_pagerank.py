import pytest

from graphrag_lab.retrieval.pagerank import personalized_pagerank

CHAIN = [("a", "b"), ("b", "c"), ("c", "d")]


def test_scores_sum_to_one():
    scores = personalized_pagerank(CHAIN, ["a"])

    assert sum(scores.values()) == pytest.approx(1.0)


def test_score_falls_with_distance_from_the_seed():
    scores = personalized_pagerank(CHAIN, ["a"])

    assert scores["a"] > scores["b"] > scores["c"] > scores["d"] > 0


def test_unreachable_nodes_get_nothing():
    scores = personalized_pagerank([*CHAIN, ("x", "y")], ["a"])

    assert scores.get("x", 0.0) == 0.0 and scores.get("y", 0.0) == 0.0


def test_symmetric_graph_gives_symmetric_scores():
    scores = personalized_pagerank([("hub", "left"), ("hub", "right")], ["hub"])

    assert scores["left"] == pytest.approx(scores["right"])


def test_node_between_two_seeds_outranks_a_node_beside_one():
    edges = [("s1", "between"), ("between", "s2"), ("s1", "beside")]

    scores = personalized_pagerank(edges, ["s1", "s2"])

    assert scores["between"] > scores["beside"]


def test_parallel_edges_pull_more_weight():
    edges = [("s", "twice"), ("s", "twice"), ("s", "once")]

    scores = personalized_pagerank(edges, ["s"])

    assert scores["twice"] == pytest.approx(2 * scores["once"])


def test_seed_without_edges_keeps_all_the_mass():
    assert personalized_pagerank(CHAIN, ["alone"]) == {"alone": pytest.approx(1.0)}


def test_no_seeds_gives_no_scores():
    assert personalized_pagerank(CHAIN, []) == {}


def test_zero_damping_never_leaves_the_seeds():
    scores = personalized_pagerank(CHAIN, ["a", "c"], damping=0.0)

    assert scores["a"] == scores["c"] == pytest.approx(0.5)
    assert scores["b"] == scores["d"] == 0.0


def test_damping_of_one_is_rejected():
    with pytest.raises(ValueError):
        personalized_pagerank(CHAIN, ["a"], damping=1.0)
