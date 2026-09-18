"""Every metric declares a valid range, and holds it at its degenerate boundaries.

A negative value reaching results.csv is what prompted this audit: `entropy` and
`explanation_diversity` returned negative zero, which is not < 0 but serialises as
"-0.000000" and reads as a negative entropy.
"""
import math

import pandas as pd
import pytest

from rexbench.metrics import accuracy as A
from rexbench.metrics import explanation_quality as EQ
from rexbench.metrics import fairness as F

# Every metric that lands in results.parquet, with the range it must declare.
DECLARED = {
    (A.ndcg_at_k, (0.0, 1.0)), (A.map_at_k, (0.0, 1.0)), (A.ap_at_k, (0.0, 1.0)),
    (A.average_recommendation_popularity, (0.0, math.inf)),
    (A.average_recommendation_popularity_normalized, (0.0, 1.0)),
    (F.gini, (0.0, 1.0)), (F.variance, (0.0, 0.5)), (F.entropy, (0.0, math.inf)),
    (F.tail_coverage_rec, (0.0, 1.0)), (F.item_coverage, (0.0, 1.0)),
    (F.item_exposure_gini, (0.0, 1.0)), (F.tail_item_coverage, (0.0, 1.0)),
    (F.tail_exposure_share, (0.0, 1.0)),
    (EQ.model_fidelity, (0.0, 1.0)), (EQ.tail_coverage, (0.0, 1.0)),
    (EQ.explanation_diversity, (0.0, math.inf)), (EQ.explanation_arp, (0.0, math.inf)),
    (EQ.explanation_coverage, (0.0, 1.0)), (EQ.explanation_personalization, (0.0, 1.0)),
}


@pytest.mark.parametrize("fn,expected", sorted(DECLARED, key=lambda t: t[0].__name__))
def test_every_reported_metric_declares_its_range(fn, expected):
    assert getattr(fn, "valid_range", None) == expected, fn.__name__


def _recs(per_user):
    return pd.DataFrame(
        [(u, i, 1.0) for u, items in per_user.items() for i in items],
        columns=["user_id", "item_id", "score"],
    )


def _exp(sets):
    frame = pd.DataFrame({
        "user_id": range(len(sets)), "item_id": range(len(sets)), "explanation_set": sets,
    })
    return EQ.annotate_tail_counts(frame, pd.Index([0, 1]))


# --------------------------------------------------------------------- no negative zeros

def test_entropy_of_a_single_item_is_positive_zero():
    value = F.entropy(_recs({0: [0]}), 1)
    assert value == 0.0
    assert math.copysign(1.0, value) > 0, "must not be -0.0 (serialises as '-0.000000')"


def test_explanation_diversity_of_one_user_is_positive_zero():
    value = EQ.explanation_diversity(_exp([{5, 6}]))
    assert value == 0.0
    assert math.copysign(1.0, value) > 0


# ------------------------------------------------------------ variance: bound and empties

def test_variance_upper_bound_is_half_not_one():
    """Maximum is reached with half the users at 0 and half at 1."""
    worst = F.variance({i: (0.0 if i < 50 else 1.0) for i in range(100)})
    assert worst == pytest.approx(0.5)


def test_variance_returns_nan_instead_of_crashing_on_empty_input():
    """Used to raise ZeroDivisionError, killing the whole (dataset, model, seed) row."""
    assert math.isnan(F.variance({}))
    assert math.isnan(F.variance({0: 0.7}))


def test_variance_is_zero_when_every_user_scores_the_same():
    assert F.variance({i: 0.5 for i in range(10)}) == 0.0


# ------------------------------------------------- degenerate inputs stay inside the range

def test_gini_reports_nan_for_undefined_cases_rather_than_a_sentinel():
    assert math.isnan(F.gini({}))
    assert math.isnan(F.gini({0: 0.5}))
    assert math.isnan(F.gini({i: 0.0 for i in range(5)}))
    assert F.gini({i: 0.4 for i in range(5)}) == 0.0


def test_item_exposure_gini_is_exactly_zero_for_perfectly_even_exposure():
    """Floating point must not push the Lorenz formula below 0 and trip the validator."""
    for n in (4, 10, 100, 1000):
        value = F.item_exposure_gini(_recs({u: [u] for u in range(n)}), n, 1)
        assert value == 0.0, n
        assert math.copysign(1.0, value) > 0


def test_item_exposure_gini_is_nan_when_nothing_was_recommended():
    empty = pd.DataFrame(columns=["user_id", "item_id", "score"])
    assert math.isnan(F.item_exposure_gini(empty, 10, 5))


def test_explanation_metrics_survive_empty_explanation_sets():
    frame = _exp([set(), set()])
    assert EQ.model_fidelity(frame) == 0.0
    assert math.isnan(EQ.tail_coverage(frame))
    assert EQ.explanation_diversity(frame) == 0.0
    assert EQ.explanation_coverage(frame, 10) == 0.0
    assert EQ.explanation_arp(frame, pd.Series([1, 2, 3])) == 0.0


def test_personalization_is_nan_for_a_single_user_and_zero_for_identical_sets():
    assert math.isnan(EQ.explanation_personalization(_exp([{1, 2}])))
    assert EQ.explanation_personalization(_exp([{1, 2}, {1, 2}])) == 0.0


def test_accuracy_metrics_handle_empty_ground_truth_and_recommendations():
    assert A.ap_at_k([1, 2, 3], set(), 10) == 0.0
    empty = pd.DataFrame(columns=["user_id", "item_id", "score"])
    assert A.average_recommendation_popularity(empty, pd.Series([1, 2, 3]), 10) == 0.0
    single = pd.DataFrame([(0, 0, 1.0)], columns=["user_id", "item_id", "score"])
    assert A.average_recommendation_popularity_normalized(single, pd.Series([5, 5, 5]), 10) == 0.0
