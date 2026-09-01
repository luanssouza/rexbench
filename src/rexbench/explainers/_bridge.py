"""Shared machinery for the AR/KNN explainer adapters: reuses
rexfair.rexfair.explanation.explainers.load_model unmodified, which needs (a) a RecBole
.pth checkpoint on disk (RecBoleModelAdapter.fit() already produces one via Trainer's own
checkpointing) and (b) a RecOXPlainer config.yml entry naming a headerless CSV — generated
here on the fly from the DatasetBundle's train split, in the exact format rexfair's own
checked-in configs/config.yml already uses for its datasets (AUDIT.md 1.4 confirmed this
format: `userId,itemId,rating,timestamp`, no header, comma-separated).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml

import rexbench.vendor  # noqa: F401
from rexfair.explanation.explainers import load_model
from rexfair.explanation.model import CustomModel
from recoxplainer.data_reader.data_reader import DataReader

from rexbench.core.dataset import DatasetBundle


def load_recoxplainer_bridge(
    checkpoint_path: str, dataset: DatasetBundle
) -> Tuple[CustomModel, pd.DataFrame, DataReader]:
    work_dir = tempfile.TemporaryDirectory(prefix=f"rexbench-recoxplainer-{dataset.name}-")
    data_csv = Path(work_dir.name) / "train.csv"
    cols = ["user_id", "item_id", "rating", "timestamp"]
    dataset.train[cols].to_csv(data_csv, index=False, header=False)

    config_yaml = Path(work_dir.name) / "config.yml"
    config_yaml.write_text(
        yaml.safe_dump(
            {
                "base": {
                    dataset.name: {
                        "filepath_or_buffer": str(data_csv),
                        "sep": ",",
                        "skiprows": 0,
                        "names": ["userId", "itemId", "rating", "timestamp"],
                    }
                }
            }
        )
    )

    custom_model, recommendations, data = load_model(
        model_path=checkpoint_path,
        data_name=dataset.name,
        recoxplainer_config_path=str(config_yaml),
        full_sort=False,
    )
    custom_model._rexbench_work_dir = work_dir  # keep temp dir alive with the returned objects
    return custom_model, recommendations, data


def explanations_to_canonical(explanations_df: pd.DataFrame, data: DataReader) -> pd.DataFrame:
    """Converts RecOXPlainer's internal userId/itemId/explanations columns back to
    rexbench's canonical ids, matching rexfair.rexfair.explanation.explainers.
    recommendations_to_original_id's logic, and renames to rexbench's schema
    (user_id/item_id/explanation_set) for metrics/explanation_quality.py."""
    rows = []
    for _, r in explanations_df.iterrows():
        rows.append(
            {
                "user_id": int(data.original_user_id.loc[r["userId"]]["user_id"]),
                "item_id": int(data.original_item_id.loc[r["itemId"]]["item_id"]),
                "explanation_set": {
                    int(data.original_item_id.loc[e]["item_id"]) for e in r["explanations"]
                },
            }
        )
    return pd.DataFrame(rows, columns=["user_id", "item_id", "explanation_set"])
