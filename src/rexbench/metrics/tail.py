"""Single source of truth for the tail-item set.

Ported from the predecessor pipeline's compute_tail_items (AUDIT.md section 1.3 /
4.5 found this implementation correct — same formula, only the column parameter is now a
name instead of a positional index since DatasetBundle always uses named columns).
Computed exactly once per dataset in core/dataset.py::build_dataset_bundle and threaded
through as an argument to every tail-based metric, per AUDIT.md's "tail item set computed
once per dataset and reused by every tail metric" requirement.
"""
from __future__ import annotations

import math

import pandas as pd


def compute_tail_items(df: pd.DataFrame, item_col: str = "item_id", tail_fraction: float = 0.20) -> pd.Index:
    frequency = df.groupby(item_col).size()
    n_tail = math.floor(frequency.shape[0] * tail_fraction)
    return frequency.sort_values(ascending=False).tail(n_tail).index
