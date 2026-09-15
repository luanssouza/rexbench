"""Item-side (provider-side) fairness metrics — see metrics/fairness.py's section header.

Expected values here are hand-computed, not snapshotted from the implementation.
"""
import math

import numpy as np
import pandas as pd
import pytest

from rexbench.metrics.fairness import (
    item_coverage, item_exposure_gini, tail_coverage_rec, tail_exposure_share,
    tail_item_coverage,
)


def _recs(rows):
    """rows: (user_id, item_id, score) — score decides rank within a user."""
    return pd.DataFrame(rows, columns=["user_id", "item_id", "score"])


# --------------------------------------------------------------------------- item_coverage

def test_item_coverage_is_distinct_items_over_catalog():
    # 3 users x top-2 -> items {0,1,2,3} distinct out of a 10-item catalog
    recs = _recs([
        (0, 0, 0.9), (0, 1, 0.8),
        (1, 0, 0.9), (1, 2, 0.7),
        (2, 3, 0.9), (2, 1, 0.5),
    ])
    assert item_coverage(recs, num_items=10, top_k=2) == pytest.approx(4 / 10)


def test_item_coverage_respects_top_k_truncation():
    # at k=1 only the top-scored item per user counts -> {0, 5} = 2 items, not 4
    recs = _recs([
        (0, 0, 0.9), (0, 1, 0.1),
        (1, 5, 0.9), (1, 6, 0.1),
    ])
    assert item_coverage(recs, num_items=10, top_k=1) == pytest.approx(2 / 10)
    assert item_coverage(recs, num_items=10, top_k=2) == pytest.approx(4 / 10)


def test_item_coverage_zero_catalog_is_zero_not_crash():
    assert item_coverage(_recs([]), num_items=0, top_k=5) == 0.0


# ---------------------------------------------------------------- item_exposure_gini

def test_item_exposure_gini_zero_when_every_item_equally_exposed():
    # 4 users, top-1, each of the 4 catalog items recommended exactly once
    recs = _recs([(u, u, 0.9) for u in range(4)])
    assert item_exposure_gini(recs, num_items=4, top_k=1) == pytest.approx(0.0, abs=1e-12)


def test_item_exposure_gini_max_when_all_exposure_on_one_item():
    # every user gets item 0; the other 9 items get nothing.
    # discrete Gini maximum for n items is (n-1)/n = 0.9 at n=10.
    recs = _recs([(u, 0, 0.9) for u in range(5)])
    assert item_exposure_gini(recs, num_items=10, top_k=1) == pytest.approx(0.9)


def test_item_exposure_gini_counts_never_recommended_items_as_zeros():
    # Same recommendations, bigger catalog -> more unexposed items -> higher inequality.
    recs = _recs([(u, u % 2, 0.9) for u in range(4)])
    small = item_exposure_gini(recs, num_items=2, top_k=1)
    large = item_exposure_gini(recs, num_items=20, top_k=1)
    assert small == pytest.approx(0.0, abs=1e-12)  # both items equally exposed
    assert large > small


def test_item_exposure_gini_nan_when_nothing_recommended():
    assert math.isnan(item_exposure_gini(_recs([]), num_items=10, top_k=5))


# ---------------------------------------------------------------- tail_item_coverage

def test_tail_item_coverage_is_distinct_tail_items_over_tail_set():
    tail = pd.Index([5, 6, 7, 8, 9])
    # tail items 5 and 6 are shown (6 twice); 7/8/9 never -> 2/5
    recs = _recs([
        (0, 0, 0.9), (0, 5, 0.8),
        (1, 6, 0.9), (1, 1, 0.8),
        (2, 6, 0.9), (2, 2, 0.8),
    ])
    assert tail_item_coverage(recs, tail, top_k=2) == pytest.approx(2 / 5)


def test_tail_item_coverage_distinguishes_same_tail_item_shown_to_everyone():
    """The exact case ACLT was introduced to catch: a high share of tail SLOTS that is
    really just one tail item repeated to every user."""
    tail = pd.Index([5, 6, 7, 8, 9])
    everyone_same = _recs([(u, 5, 0.9) for u in range(5)])
    spread_out = _recs([(u, 5 + u, 0.9) for u in range(5)])
    assert tail_item_coverage(everyone_same, tail, top_k=1) == pytest.approx(1 / 5)
    assert tail_item_coverage(spread_out, tail, top_k=1) == pytest.approx(5 / 5)


def test_tail_item_coverage_empty_tail_set_is_zero():
    assert tail_item_coverage(_recs([(0, 1, 0.5)]), pd.Index([]), top_k=5) == 0.0


# --------------------------------------------------------------- tail_exposure_share

def test_tail_exposure_share_weights_rank_1_above_rank_2():
    """One tail item + one head item per user, position swapped. Discount is
    1/log2(1+rank): rank1 -> 1.0, rank2 -> 1/log2(3) = 0.63093."""
    w1, w2 = 1.0, 1.0 / math.log2(3.0)
    tail = pd.Index([0])

    tail_first = _recs([(0, 0, 0.9), (0, 1, 0.1)])
    tail_second = _recs([(0, 1, 0.9), (0, 0, 0.1)])

    assert tail_exposure_share(tail_first, tail, top_k=2) == pytest.approx(w1 / (w1 + w2))
    assert tail_exposure_share(tail_second, tail, top_k=2) == pytest.approx(w2 / (w1 + w2))
    # burying the tail item strictly lowers the score, which is the whole point
    assert tail_exposure_share(tail_second, tail, top_k=2) < tail_exposure_share(tail_first, tail, top_k=2)


def test_tail_exposure_share_is_one_when_every_slot_is_tail():
    tail = pd.Index([0, 1])
    recs = _recs([(0, 0, 0.9), (0, 1, 0.8)])
    assert tail_exposure_share(recs, tail, top_k=2) == pytest.approx(1.0)


def test_tail_exposure_share_is_zero_when_no_tail_item_recommended():
    recs = _recs([(0, 3, 0.9), (0, 4, 0.8)])
    assert tail_exposure_share(recs, pd.Index([0, 1]), top_k=2) == 0.0


def test_tail_exposure_share_differs_from_unweighted_share():
    """Same 50% tail SLOT share in both lists, different exposure share — this is the
    information tail_coverage_rec cannot express."""
    tail = pd.Index([0])
    tail_first = _recs([(0, 0, 0.9), (0, 1, 0.1)])
    tail_second = _recs([(0, 1, 0.9), (0, 0, 0.1)])
    assert tail_exposure_share(tail_first, tail, 2) != pytest.approx(0.5)
    assert tail_exposure_share(tail_second, tail, 2) != pytest.approx(0.5)


# ------------------------------------------------- tail_coverage_rec (APLT), after the fix

def test_tail_coverage_rec_is_k_aware():
    """Regression: this used to score the whole predictions_df, so @1 and @2 came out
    identical and the k column in results.parquet was meaningless for this metric."""
    tail = pd.Index([9])
    recs = _recs([(0, 3, 0.9), (0, 9, 0.5)])  # head at rank 1, tail at rank 2
    assert tail_coverage_rec(recs, tail, top_k=1) == pytest.approx(0.0)
    assert tail_coverage_rec(recs, tail, top_k=2) == pytest.approx(0.5)


def test_tail_coverage_rec_averages_per_user_not_pooled_rows():
    """Regression: pooling every user's rows into one ratio weighted users with longer
    lists more heavily. APLT averages per-user percentages instead.

    User 0: 4 items, none tail -> 0/4. User 1: 1 item, tail -> 1/1.
    Pooled (old, wrong): 1 tail row / 5 rows      = 0.2
    Per-user (APLT):     (0/4 + 1/1) / 2 users    = 0.5
    """
    tail = pd.Index([9])
    recs = _recs([
        (0, 0, 0.9), (0, 1, 0.8), (0, 2, 0.7), (0, 3, 0.6),
        (1, 9, 0.9),
    ])
    assert tail_coverage_rec(recs, tail, top_k=4) == pytest.approx(0.5)


def test_tail_coverage_rec_matches_hand_computed_value():
    tail = pd.Index([8, 9])
    recs = _recs([
        (0, 8, 0.9), (0, 0, 0.8),   # 1/2 tail
        (1, 9, 0.9), (1, 8, 0.8),   # 2/2 tail
        (2, 0, 0.9), (2, 1, 0.8),   # 0/2 tail
    ])
    assert tail_coverage_rec(recs, tail, top_k=2) == pytest.approx((0.5 + 1.0 + 0.0) / 3)


def test_tail_coverage_rec_empty_predictions_is_zero():
    assert tail_coverage_rec(_recs([]), pd.Index([1]), top_k=5) == 0.0


def test_tail_coverage_rec_and_tail_item_coverage_answer_different_questions():
    """Same tail item shown to all 3 users: every slot-share is perfect, but only one
    distinct tail item out of 4 is actually covered."""
    tail = pd.Index([6, 7, 8, 9])
    recs = _recs([(u, 6, 0.9) for u in range(3)])
    assert tail_coverage_rec(recs, tail, top_k=1) == pytest.approx(1.0)
    assert tail_item_coverage(recs, tail, top_k=1) == pytest.approx(1 / 4)
