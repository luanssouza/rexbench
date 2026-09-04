import math

import pandas as pd
import pytest

from rexbench.core.dataset import DatasetBundle, InteractionStats, compute_interaction_stats
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
