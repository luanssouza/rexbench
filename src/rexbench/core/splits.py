"""Train/val/test split strategies.

Ported from the predecessor pipeline's preprocessing code (see AUDIT.md sections 1.4/4.2 —
found correct, reused verbatim) so rexbench has no runtime dependency outside this repo.
"""
from __future__ import annotations

import math
from typing import Tuple

import pandas as pd


def random_split(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_state: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame randomly into train, validation, and test sets.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    val_test_size = val_size + test_size
    train_df = df.sample(frac=1 - val_test_size, random_state=random_state)
    val_test_df = df.drop(train_df.index).copy()
    val_df = val_test_df.sample(frac=val_size / val_test_size, random_state=random_state)
    test_df = val_test_df.drop(val_df.index)
    return train_df, val_df, test_df


def temporal_split(
    df: pd.DataFrame,
    user_col=0,
    temp_col=1,
    val_size: float = 0.1,
    test_size: float = 0.2,
    min_interactions: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame per-user using temporal ordering.

    The most recent interactions per user are used for test and validation. Users with
    fewer than `min_interactions` rows are kept entirely in train.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    val_test_ratio = val_size + test_size
    ordered_df = df.sort_values([user_col, temp_col]).reset_index(drop=True)

    train_dfs, val_dfs, test_dfs = [], [], []

    for _u, g in ordered_df.groupby(user_col):
        if g.shape[0] < min_interactions:
            train_dfs.append(g)
            continue

        val_test_n = math.floor(g.shape[0] * val_test_ratio)
        if g.shape[0] < val_test_n:
            train_dfs.append(g)
            continue

        val_test_df = g.tail(val_test_n)
        train_dfs.append(g.drop(val_test_df.index))

        val_n = val_test_df.shape[0] * (val_size / val_test_ratio)
        if val_test_df.shape[0] < val_n:
            train_dfs.append(val_test_df)
            continue

        val_df = val_test_df.head(math.floor(val_n))
        test_df = val_test_df.drop(val_df.index)
        val_dfs.append(val_df)
        test_dfs.append(test_df)

    return pd.concat(train_dfs), pd.concat(val_dfs), pd.concat(test_dfs)