import pytest

from graphrag_lab.benchmark.metrics import (
    answer_coverage,
    exact_match,
    normalize_answer,
    recall,
    token_f1,
)


def test_normalization_drops_case_punctuation_and_articles():
    assert normalize_answer("  The Eiffel Tower, Paris! ") == "eiffel tower paris"


def test_exact_match_accepts_any_gold_answer():
    assert exact_match("the United States", ["USA", "United States"]) == 1.0
    assert exact_match("Canada", ["USA", "United States"]) == 0.0


def test_token_f1_rewards_partial_overlap():
    # 2 of the 3 predicted tokens are right, and they cover both gold tokens.
    assert token_f1("president Barack Obama", ["Barack Obama"]) == pytest.approx(0.8)


def test_token_f1_takes_the_best_gold_answer():
    assert token_f1("Obama", ["Barack Obama", "Obama"]) == 1.0


def test_token_f1_counts_repeated_tokens_once_each():
    assert token_f1("New New York", ["New York"]) == pytest.approx(0.8)


def test_token_f1_with_nothing_in_common_is_zero():
    assert token_f1("Paris", ["London"]) == 0.0
    assert token_f1("", ["London"]) == 0.0


def test_answer_coverage_needs_the_whole_answer_on_word_boundaries():
    context = "Marie Curie was born in Warsaw.\nWarsaw is in Poland."

    assert answer_coverage(context, ["Poland"]) == 1.0
    assert answer_coverage(context, ["Pol"]) == 0.0
    assert answer_coverage(context, ["France", "the Warsaw"]) == 1.0
    assert answer_coverage(context, ["France"]) == 0.0


def test_recall_is_the_share_of_gold_items_found():
    assert recall(["f1", "f3", "other"], ["f1", "f2"]) == 0.5


def test_recall_without_gold_items_is_an_error():
    with pytest.raises(ValueError):
        recall(["f1"], [])
