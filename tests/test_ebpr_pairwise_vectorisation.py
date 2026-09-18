"""avg_pairwise_similarity: the vectorised extraction must be exactly equivalent.

The old code re-scanned the whole frame once per user — O(users^2), measured at users^1.95,
which projected to 562 h for a single electronics fit. The replacement must produce a
byte-identical list of lists, including group order and within-group row order.
"""
import numpy as np
import pandas as pd
import pytest


def _old(full, col):
    users = list(dict.fromkeys(list(full["user"])))
    return [list(full.loc[full["user"] == u][col]) for u in users]


def _new(full, col):
    return list(full.groupby("user", sort=False)[col].apply(list))


def _frame(rows):
    return pd.DataFrame(rows, columns=["user", "item"])


def test_identical_on_simple_frame():
    full = _frame([(0, 5), (0, 7), (1, 2), (1, 9), (2, 3)])
    assert _old(full, "item") == _new(full, "item")


def test_preserves_first_appearance_group_order_not_sorted_order():
    """Users appear 2, 0, 1 — the result must follow that order, not 0, 1, 2."""
    full = _frame([(2, 5), (0, 7), (1, 2), (2, 8), (0, 1)])
    assert _old(full, "item") == _new(full, "item")
    assert _new(full, "item")[0] == [5, 8], "user 2 first"


def test_preserves_within_group_row_order():
    full = _frame([(0, 9), (0, 1), (0, 5)])
    assert _new(full, "item") == [[9, 1, 5]]
    assert _old(full, "item") == _new(full, "item")


def test_identical_with_duplicate_items_for_a_user():
    full = _frame([(0, 4), (0, 4), (0, 6)])
    assert _old(full, "item") == _new(full, "item")


def test_identical_for_single_item_users():
    """Under loo_eval=True these are NOT filtered out — preserved deliberately."""
    full = _frame([(0, 1), (1, 2), (1, 3)])
    assert _old(full, "item") == _new(full, "item")
    assert [1] in _new(full, "item")


def test_identical_on_randomised_frames():
    rng = np.random.default_rng(0)
    for _ in range(25):
        n_users = int(rng.integers(1, 40))
        rows = []
        for u in rng.permutation(n_users):
            for _ in range(int(rng.integers(1, 12))):
                rows.append((int(u), int(rng.integers(0, 50))))
        rng.shuffle(rows)
        full = _frame(rows)
        assert _old(full, "item") == _new(full, "item")


def test_end_to_end_metric_value_is_unchanged():
    """Full avg_pairwise_similarity result, old extraction vs new, on the same data."""
    from itertools import combinations

    rng = np.random.default_rng(7)
    sim = rng.random((60, 60))
    rows = []
    for u in range(30):
        for i in rng.choice(60, size=10, replace=False):
            rows.append((u, int(i)))
    full = _frame(rows)

    def metric(extract):
        per_user = [set(combinations(r, 2)) for r in extract(full, "item")]
        return np.mean([np.mean([sim[i, j] for (i, j) in c]) for c in per_user])

    assert metric(_old) == pytest.approx(metric(_new))
