import math

import pandas as pd
import pytest

from rexbench.core.dataset import (
    DatasetBundle, InteractionStats, _cap_interactions_per_user, _subsample_users,
    compute_interaction_stats,
)
from rexbench.metrics.tail import compute_tail_items


def test_compute_tail_items_bottom_20_percent_by_frequency():
    # 10 items, item 0 has 10 interactions down to item 9 with 1 -> bottom 20% (2 items)
    # by count are items 8 and 9.
    rows = []
    for item_id in range(10):
        count = 10 - item_id
        rows += [{"item_id": item_id} for _ in range(count)]
    df = pd.DataFrame(rows)

    tail = compute_tail_items(df, item_col="item_id", tail_fraction=0.20)
    assert set(tail) == {8, 9}


def test_compute_interaction_stats():
    df = pd.DataFrame({
        "user_id": [0, 0, 0, 1, 2],
        "item_id": [0, 1, 2, 0, 1],
        "rating": [1, 1, 1, 1, 1],
        "timestamp": [0, 0, 0, 0, 0],
    })
    stats = compute_interaction_stats(df)
    assert stats.num_users == 3
    assert stats.num_items == 3
    assert stats.num_interactions == 5
    # user 0 has 3, users 1 and 2 have 1 each -> 2/3 have fewer than 2
    assert stats.fraction_users_lt_2 == pytest.approx(2 / 3)
    assert stats.mean_interactions_per_user == pytest.approx(5 / 3)
    assert stats.median_interactions_per_user == 1.0


def _minimal_bundle(val_df: pd.DataFrame, test_df: pd.DataFrame) -> DatasetBundle:
    stats = InteractionStats(0.0, 0.0, 0.0, 0, 0, 0)
    return DatasetBundle(
        name="t", train=pd.DataFrame(columns=["user_id", "item_id", "rating", "timestamp"]),
        val=val_df, test=test_df, num_users=0, num_items=0,
        tail_items=pd.Index([]), popularity=pd.Series(dtype=int),
        user_id_map={}, item_id_map={}, kg_status="unsupported", kg_path=None,
        interaction_stats=stats, topk=[10], tail_fraction=0.2,
    )


def test_user_val_dict_and_user_test_dict_are_independent():
    val_df = pd.DataFrame({"user_id": [0, 0, 1], "item_id": [10, 11, 12]})
    test_df = pd.DataFrame({"user_id": [0, 1], "item_id": [20, 21]})
    bundle = _minimal_bundle(val_df, test_df)

    assert bundle.user_val_dict() == {0: {10, 11}, 1: {12}}
    assert bundle.user_test_dict() == {0: {20}, 1: {21}}


def _dense_df(num_users: int, interactions_per_user: int) -> pd.DataFrame:
    rows = []
    for u in range(num_users):
        for i in range(interactions_per_user):
            # each user's items are their own private range, so per-user density is
            # unambiguous to check after subsampling, and the item catalog size is
            # exactly num_users * interactions_per_user before any subsampling.
            rows.append({"user_id": u, "item_id": u * interactions_per_user + i})
    return pd.DataFrame(rows)


def test_subsample_users_keeps_every_interaction_of_kept_users():
    df = _dense_df(num_users=50, interactions_per_user=20)
    sampled = _subsample_users(df, max_users=5, seed=1)

    assert sampled["user_id"].nunique() == 5
    # every kept user still has all 20 of their original interactions -- density per user
    # is preserved, not diluted the way a random row sample would dilute it.
    counts = sampled.groupby("user_id").size()
    assert (counts == 20).all()


def test_subsample_users_shrinks_the_item_catalog():
    # 50 users x 20 items each = 1000 distinct items before sampling (private ranges).
    df = _dense_df(num_users=50, interactions_per_user=20)
    sampled = _subsample_users(df, max_users=5, seed=1)
    # only the 5 kept users' 20-item private ranges survive -> at most 100 distinct items,
    # not 1000 -- this is the mechanism that makes ambar/lastfm1k's huge catalogs testable.
    assert sampled["item_id"].nunique() == 100


def test_subsample_users_deterministic_given_same_seed():
    df = _dense_df(num_users=50, interactions_per_user=20)
    a = _subsample_users(df, max_users=5, seed=42)
    b = _subsample_users(df, max_users=5, seed=42)
    assert set(a["user_id"].unique()) == set(b["user_id"].unique())


def test_subsample_users_noop_when_max_users_exceeds_actual():
    df = _dense_df(num_users=10, interactions_per_user=3)
    sampled = _subsample_users(df, max_users=1000, seed=1)
    assert len(sampled) == len(df)


def test_cap_interactions_per_user_bounds_rows_and_catalog():
    # one very active user (2000 private-range items) plus a few normal ones -- this is the
    # lastfm1k-shaped case: _subsample_users alone wouldn't bound the catalog here.
    df = _dense_df(num_users=5, interactions_per_user=2000)
    capped = _cap_interactions_per_user(df, max_interactions_per_user=10, seed=1)

    counts = capped.groupby("user_id").size()
    assert (counts == 10).all()
    assert capped["item_id"].nunique() <= 5 * 10  # bounded by users x cap, not the original 10000


def test_cap_interactions_per_user_leaves_under_cap_users_untouched():
    df = _dense_df(num_users=3, interactions_per_user=4)
    capped = _cap_interactions_per_user(df, max_interactions_per_user=100, seed=1)
    assert len(capped) == len(df)


def test_cap_interactions_per_user_deterministic_given_same_seed():
    df = _dense_df(num_users=5, interactions_per_user=200)
    a = _cap_interactions_per_user(df, max_interactions_per_user=10, seed=7)
    b = _cap_interactions_per_user(df, max_interactions_per_user=10, seed=7)
    pd.testing.assert_frame_equal(
        a.sort_values(["user_id", "item_id"]).reset_index(drop=True),
        b.sort_values(["user_id", "item_id"]).reset_index(drop=True),
    )
