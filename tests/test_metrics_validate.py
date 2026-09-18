import math

import pytest

from rexbench.metrics.fairness import gini, variance
from rexbench.metrics.validate import MetricRangeError, validated_range


def test_validated_range_raises_on_out_of_range():
    @validated_range(0.0, 1.0)
    def bad_metric():
        return -1.0

    with pytest.raises(MetricRangeError):
        bad_metric()


def test_validated_range_allows_nan_when_permitted():
    @validated_range(0.0, 1.0, allow_nan=True)
    def undefined_metric():
        return float("nan")

    assert math.isnan(undefined_metric())


def test_validated_range_rejects_nan_by_default():
    @validated_range(0.0, 1.0)
    def undefined_metric():
        return float("nan")

    with pytest.raises(MetricRangeError):
        undefined_metric()


# AUDIT.md section 5.3's exact traced inputs for the original -1.0 sentinel bug.
def test_gini_all_zero_scores_is_nan_not_negative_one():
    result = gini({1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0})
    assert math.isnan(result)


def test_gini_single_user_is_nan():
    result = gini({1: 0.37})
    assert math.isnan(result)


def test_gini_all_equal_nonzero_is_zero_not_negative_one():
    result = gini({1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5})
    assert result == 0.0


def test_gini_normal_case_unchanged():
    result = gini({1: 0.1, 2: 0.9})
    assert result == pytest.approx(0.4)


def test_gini_never_returns_out_of_range_value():
    # exercised via the decorator itself: any [0,1]-violating or non-NaN-invalid return
    # would raise inside gini() before this line completes.
    for scores in ({1: 1.0, 2: 2.0, 3: 3.0}, {1: 0.0, 2: 5.0}, {i: float(i) for i in range(10)}):
        value = gini(scores)
        assert math.isnan(value) or 0.0 <= value <= 1.0


def test_variance_unchanged_formula():
    """Same formula as always — mean squared pairwise difference over n^2 — but exercised
    inside variance's documented domain (per-user NDCG, so scores in [0,1]).

    It used to pass {1: 1.0, 2: 3.0}, which is outside that domain and yields 2.0. That went
    unnoticed while variance declared no range; now that it declares [0, 0.5] (the true
    maximum for [0,1] inputs, reached with half the users at 0 and half at 1) the old inputs
    correctly trip MetricRangeError. The formula itself is unchanged: for {0.0, 1.0} the
    ordered pairs give (0-1)^2 + (1-0)^2 = 2, over n^2 = 4, i.e. 0.5.
    """
    assert variance({1: 0.0, 2: 1.0}) == pytest.approx(0.5)
    assert variance({1: 0.25, 2: 0.75}) == pytest.approx(2 * 0.5 ** 2 / 4)  # diff is 0.5
