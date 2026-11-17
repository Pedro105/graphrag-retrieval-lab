import pytest

from graphrag_lab.benchmark.stats import bootstrap_ci, paired_permutation_test


def test_interval_brackets_the_mean():
    scores = [0.0, 1.0] * 50

    mean, low, high = bootstrap_ci(scores, resamples=2000)

    assert mean == 0.5
    assert 0.35 < low < mean < high < 0.65


def test_interval_narrows_with_more_data():
    _, small_low, small_high = bootstrap_ci([0.0, 1.0] * 20, resamples=2000)
    _, large_low, large_high = bootstrap_ci([0.0, 1.0] * 500, resamples=2000)

    assert large_high - large_low < small_high - small_low


def test_constant_scores_give_a_zero_width_interval():
    assert bootstrap_ci([0.7] * 30, resamples=500) == pytest.approx((0.7, 0.7, 0.7))


def test_same_seed_same_interval():
    scores = [0.1, 0.4, 0.4, 0.9, 1.0, 0.0, 0.6]

    assert bootstrap_ci(scores, seed=3) == bootstrap_ci(scores, seed=3)


def test_a_consistent_improvement_is_significant():
    baseline = [0.0, 1.0] * 20
    better = [min(1.0, score + 0.3) if i % 4 else score for i, score in enumerate(baseline)]

    assert paired_permutation_test(better, baseline) < 0.01


def test_identical_systems_are_not_significant():
    scores = [0.2, 0.8, 0.5, 1.0, 0.0]

    assert paired_permutation_test(scores, scores) == 1.0


def test_differences_that_cancel_out_are_not_significant():
    a = [1.0, 0.0] * 20
    b = [0.0, 1.0] * 20

    assert paired_permutation_test(a, b) > 0.5


def test_unpaired_inputs_are_rejected():
    with pytest.raises(ValueError):
        paired_permutation_test([1.0, 0.0], [1.0])
