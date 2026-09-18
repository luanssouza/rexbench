"""Every model must train on the same rows and be selected under a valid protocol.

Two confounds are fixed here: PGPR was trained on train+val while the others got train
alone, and EBPR's leave-one-out selector was fed a random multi-item validation split.
"""
import pandas as pd
import pytest

from rexbench.models.ebpr_adapter import _leave_one_out


def _split(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


# ------------------------------------------------- EBPR selection split is true leave-one-out

def test_reduces_to_exactly_one_row_per_user():
    split = _split([
        (0, 10, 1.0, 100), (0, 11, 1.0, 300), (0, 12, 1.0, 200),
        (1, 20, 1.0, 50), (1, 21, 1.0, 70),
        (2, 30, 1.0, 10),
    ])
    out = _leave_one_out(split)
    assert len(out) == 3
    assert out["user_id"].tolist() == [0, 1, 2] or sorted(out["user_id"]) == [0, 1, 2]
    assert out["user_id"].is_unique


def test_keeps_the_most_recent_interaction_per_user():
    """Matches EBPR's own convention: _split_loo uses rank_latest == 1."""
    split = _split([
        (0, 10, 1.0, 100), (0, 11, 1.0, 300), (0, 12, 1.0, 200),
        (1, 20, 1.0, 50), (1, 21, 1.0, 70),
    ])
    out = _leave_one_out(split).set_index("user_id")["item_id"].to_dict()
    assert out == {0: 11, 1: 21}


def test_single_interaction_users_are_preserved():
    split = _split([(0, 10, 1.0, 5)])
    assert _leave_one_out(split)["item_id"].tolist() == [10]


def test_empty_split_stays_empty():
    assert _leave_one_out(_split([])).empty


def test_is_idempotent():
    split = _split([(0, 10, 1.0, 1), (0, 11, 1.0, 2), (1, 20, 1.0, 3)])
    once = _leave_one_out(split)
    pd.testing.assert_frame_equal(once, _leave_one_out(once))


def test_collapses_the_row_multiplication_that_slowed_evaluate():
    """metrics.py merges the held-out frame on 'user', so k items per user multiply that
    user's rows k-fold. Measured 12x on ml100k and 19.5x on ml1m before this reduction."""
    rows = [(u, 100 + i, 1.0, i) for u in range(50) for i in range(12)]
    split = _split(rows)
    assert len(split) == 600
    reduced = _leave_one_out(split)
    assert len(reduced) == 50
    # the merge blow-up is quadratic in items-per-user; 1 item per user removes it entirely
    assert reduced.groupby("user_id").size().max() == 1


def test_preserves_the_canonical_columns():
    split = _split([(0, 10, 4.0, 7)])
    assert list(_leave_one_out(split).columns) == ["user_id", "item_id", "rating", "timestamp"]


# ------------------------------------------------------ PGPR trains on the same rows as the rest

def test_pgpr_materialises_train_only_not_train_plus_val():
    """Reads the adapter source: folding val into PGPR's training set gave it ~10% more
    interactions than EBPR/RecBole, confounding any comparison."""
    import inspect

    from rexbench.models import pgpr_adapter

    source = inspect.getsource(pgpr_adapter.PGPRModelAdapter.fit)
    assert "train_df=dataset.train," in source
    assert "pd.concat([dataset.train, dataset.val]" not in source
