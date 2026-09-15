"""Iterative k-core filtering (FilterConfig) — the mechanism that makes lastfm1k runnable.

See README's "LastFM1K memory" section for why this exists.
"""
import pandas as pd
import pytest

from rexbench.config.schema import DatasetConfig, FilterConfig
from rexbench.core.dataset import _apply_kcore_filter, _split_settings


def _df(pairs):
    return pd.DataFrame(
        [(u, i, 1.0, 0) for u, i in pairs],
        columns=["user_id", "item_id", "rating", "timestamp"],
    )


def test_no_filtering_by_default_returns_frame_unchanged():
    df = _df([(0, 0), (1, 1)])
    out = _apply_kcore_filter(df, FilterConfig())
    pd.testing.assert_frame_equal(out, df)


def test_drops_items_below_item_threshold():
    # item 0 appears 3x, item 9 once -> only item 9 is dropped
    df = _df([(0, 0), (1, 0), (2, 0), (3, 9)])
    out = _apply_kcore_filter(df, FilterConfig(min_item_interactions=3))
    assert set(out["item_id"]) == {0}
    assert len(out) == 3


def test_drops_users_below_user_threshold():
    df = _df([(0, 0), (0, 1), (0, 2), (1, 0)])
    out = _apply_kcore_filter(df, FilterConfig(min_user_interactions=3))
    assert set(out["user_id"]) == {0}


def test_iterates_because_dropping_items_can_drop_users():
    """The case a single pass gets wrong. User 1 has 2 interactions, so it survives a
    user-threshold pass — but one of them is with a singleton item, and once that item is
    removed user 1 falls to 1 interaction and must go too."""
    df = _df([
        (0, 0), (1, 0),          # item 0: 2 interactions, survives
        (0, 1), (1, 1),          # item 1: 2 interactions, survives
        (0, 2), (1, 99),         # item 2 and item 99: 1 each, both dropped
    ])
    out = _apply_kcore_filter(
        df, FilterConfig(min_item_interactions=2, min_user_interactions=3)
    )
    # user 0 keeps items {0,1} = 2 < 3 -> dropped as well; everything goes
    assert len(out) == 0


def test_converges_and_is_stable_on_already_conforming_data():
    # 3 users x 3 items, fully dense: nothing is below any threshold of 3
    df = _df([(u, i) for u in range(3) for i in range(3)])
    out = _apply_kcore_filter(
        df, FilterConfig(min_item_interactions=3, min_user_interactions=3)
    )
    assert len(out) == 9


def test_max_iterations_is_respected():
    df = _df([(u, u) for u in range(50)])  # every item/user has exactly 1
    out = _apply_kcore_filter(
        df, FilterConfig(min_item_interactions=2, max_iterations=1)
    )
    assert len(out) == 0  # one pass is enough here


def test_filter_is_part_of_the_persisted_split_fingerprint():
    """A split materialized under one filter must not be silently reused under another —
    the filter changes which rows exist at all."""
    base = DatasetConfig(name="d", loader="ml100k", raw_path="x")
    filtered = base.model_copy(
        update={"filter": FilterConfig(min_item_interactions=20)}
    )
    assert _split_settings(base) != _split_settings(filtered)
    assert _split_settings(base)["filter"]["min_item_interactions"] is None
    assert _split_settings(filtered)["filter"]["min_item_interactions"] == 20
