"""Every algorithm must be scored under exactly the same protocol.

Three defects broke that premise and are fixed here; these tests pin the corrected
behaviour. See metrics/accuracy.py and models/pgpr_adapter.py for the full write-ups.
"""
import math

import numpy as np
import pandas as pd
import pytest

from rexbench.metrics.accuracy import ap_at_k, dataset_ndcg_k, map_at_k, ndcg_at_k


def _recs(per_user):
    return pd.DataFrame(
        [(u, i, r + 1, 1.0 / (r + 1)) for u, items in per_user.items() for r, i in enumerate(items)],
        columns=["user_id", "item_id", "rank", "score"],
    )


# ------------------------------------------------- same evaluated population for every model

def test_short_lists_no_longer_drop_users_from_the_average():
    """A model returning fewer than k items must be scored on the same users as one that
    always returns k — otherwise the two averages describe different populations."""
    ground_truth = {u: {100 + u} for u in range(5)}
    long_lists = {u: [100 + u] + [200 + j for j in range(9)] for u in range(5)}
    short_lists = {
        u: ([100 + u] + [200 + j for j in range(9)] if u < 2 else [100 + u, 201, 202])
        for u in range(5)
    }
    assert len(dataset_ndcg_k(_recs(long_lists), ground_truth, 10)) == 5
    assert len(dataset_ndcg_k(_recs(short_lists), ground_truth, 10)) == 5


def test_user_with_no_recommendations_scores_zero_rather_than_vanishing():
    ground_truth = {0: {7}, 1: {8}}
    scores = dataset_ndcg_k(_recs({0: [7, 1, 2]}), ground_truth, 10)
    assert set(scores) == {0, 1}
    assert scores[1] == 0.0


def test_map_uses_the_same_population_as_ndcg():
    ground_truth = {u: {100 + u} for u in range(4)}
    per_user = {0: [100, 1, 2], 1: [101], 2: [999], 3: [103]}
    recs = _recs(per_user)
    assert len(dataset_ndcg_k(recs, ground_truth, 10)) == len(ground_truth)
    assert 0.0 < map_at_k(recs, ground_truth, 10) <= 1.0


# ------------------------------------------------------------- NDCG normalised by the ideal

def test_ndcg_normalises_by_the_ideal_not_by_the_hits_found():
    """2 hits out of 20 relevant items, both at the top. The old formula returned 1.0."""
    relevant = set(range(20))
    topk = [0, 1] + [900 + j for j in range(8)]
    hits = [1 if i in relevant else 0 for i in topk]

    # Expectation is built from the module's own dcg_at_k so the test pins the
    # NORMALISATION change, not the discount convention (dcg_at_k uses method=0).
    from rexbench.metrics.accuracy import dcg_at_k

    value = ndcg_at_k(hits, 10, n_relevant=len(relevant))
    ideal = [1] * 10
    assert value == pytest.approx(dcg_at_k(hits, 10, 0) / dcg_at_k(ideal, 10, 0))
    assert value < 0.4, "must no longer report a perfect score for 2/20 recall"
    assert ndcg_at_k(hits, 10) == pytest.approx(1.0), "the old hits-only normalisation"


def test_ndcg_is_one_only_when_every_relevant_item_is_found_and_ranked_first():
    relevant = {0, 1, 2}
    perfect = [1 if i in relevant else 0 for i in [0, 1, 2] + [900 + j for j in range(7)]]
    assert ndcg_at_k(perfect, 10, n_relevant=3) == pytest.approx(1.0)


def test_ndcg_rewards_finding_more_relevant_items():
    relevant = set(range(10))
    few = [1, 1] + [0] * 8
    many = [1] * 6 + [0] * 4
    assert ndcg_at_k(many, 10, n_relevant=len(relevant)) > ndcg_at_k(few, 10, n_relevant=len(relevant))


def test_ndcg_still_rewards_ranking_hits_higher():
    relevant = set(range(10))
    early = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    late = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    assert ndcg_at_k(early, 10, n_relevant=len(relevant)) > ndcg_at_k(late, 10, n_relevant=len(relevant))


def test_ndcg_and_map_now_agree_on_a_perfect_ranking():
    """They used to disagree: MAP normalised by min(|rel|, k), NDCG by the hits found."""
    relevant = {0, 1, 2}
    topk = [0, 1, 2] + [900 + j for j in range(7)]
    hits = [1 if i in relevant else 0 for i in topk]
    assert ndcg_at_k(hits, 10, n_relevant=3) == pytest.approx(1.0)
    assert ap_at_k(topk, relevant, 10) == pytest.approx(1.0)


def test_ndcg_stays_within_range_for_short_lists():
    """A user cannot have more hits than relevant items, so got <= n_rel."""
    for n_rel in (1, 5, 20):
        for got in (0, 1, 3):
            if got > n_rel:
                continue
            hits = [1] * got
            assert 0.0 <= ndcg_at_k(hits, 10, n_relevant=n_rel) <= 1.0


# ------------------------------------------------------- PGPR returns k DISTINCT items

def _dedup(by_user_entries, k):
    """Mirrors the de-duplication in PGPRModelAdapter.recommend()."""
    best = {}
    for prob, item_id, path in by_user_entries:
        if item_id not in best or prob > best[item_id][0]:
            best[item_id] = (prob, item_id, path)
    return [i for _, i, _ in sorted(best.values(), key=lambda t: -t[0])[:k]]


def test_pgpr_recommendations_are_deduplicated_by_item():
    entries = [(0.9, 42, None), (0.8, 42, None), (0.7, 42, None), (0.6, 7, None), (0.5, 7, None)]
    assert _dedup(entries, 5) == [42, 7]


def test_pgpr_dedup_keeps_the_highest_probability_path_per_item():
    entries = [(0.3, 1, "lo"), (0.95, 1, "hi"), (0.5, 2, "x")]
    best = {}
    for prob, item_id, path in entries:
        if item_id not in best or prob > best[item_id][0]:
            best[item_id] = (prob, item_id, path)
    assert best[1][2] == "hi"


def test_pgpr_dedup_preserves_probability_ordering():
    entries = [(0.1, 5, None), (0.9, 3, None), (0.5, 8, None)]
    assert _dedup(entries, 3) == [3, 8, 5]
