"""Shared machinery for the AR/KNN explainer adapters.

CustomModel, _PatchedDataReader, load_model, and post_hoc_ar are ported from the
predecessor pipeline's explanation layer (AUDIT.md 1.2 found these correct, reused
verbatim — same recoxplainer.explain.ARPostHocExplainer under the hood via post_hoc_ar).

load_model() needs (a) a RecBole .pth checkpoint on disk (RecBoleModelAdapter.fit() already
produces one via Trainer's own checkpointing) and (b) a RecOXPlainer config.yml entry naming
a headerless CSV — generated here on the fly from the DatasetBundle's train split, in the
format `userId,itemId,rating,timestamp`, no header, comma-separated (AUDIT.md 1.4).
"""
from __future__ import annotations

import logging
import tempfile
import warnings
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
import yaml

import rexbench.vendor  # noqa: F401  (installs the recoxplainer submodule sys.path shim)
from recbole.data.interaction import Interaction
from recbole.quick_start.quick_start import load_data_and_model
from recoxplainer.config import load_config
from recoxplainer.data_reader.data_reader import DataReader
from recoxplainer.explain import ARPostHocExplainer
from recoxplainer.recommender import Recommender

from rexbench.core.dataset import DatasetBundle


class CustomModel(torch.nn.Module):
    """Wraps a trained RecBole model so it can be used with RecOXPlainer.

    RecOXPlainer expects `predict`/`full_sort_predict` methods operating on RecOXPlainer-
    internal IDs. This translates those IDs to RecBole tokens before forwarding the call.
    """

    def __init__(self, recbole_model, recbole_dataset, recoxplainer_dataset, batch_size: int = 2048):
        super().__init__()
        self.recbole_model = recbole_model
        self.recbole_dataset = recbole_dataset
        self.recoxplainer_dataset = recoxplainer_dataset
        self.batch_size = batch_size
        self.batch_user_size = batch_size
        self.device = recbole_model.device

    def predict(self, user_id, item_id):
        uid = self.recoxplainer_dataset.original_user_id.loc[user_id]["user_id"]
        iid = [self.recoxplainer_dataset.original_item_id.loc[i]["item_id"] for i in item_id]

        uids = self.recbole_dataset.token2id(self.recbole_dataset.uid_field, [str(uid)])
        iids = self.recbole_dataset.token2id(self.recbole_dataset.iid_field, [str(i) for i in iid])

        model_name = str(self.recbole_model).split("(")[0]
        if "BPR" in model_name or "SLIME" in model_name:
            inter = Interaction({
                "user_id": torch.tensor(uids, device=self.device),
                "item_id": torch.tensor(iids, device=self.device),
            })
            pred = self.recbole_model.predict(inter)
        else:
            pred = torch.Tensor([
                self.recbole_model.predict(Interaction({
                    "user_id": torch.tensor(uids, device=self.device),
                    "item_id": torch.tensor([i], device=self.device),
                }))
                for i in iids
            ])
        return pred.cpu().tolist()

    def full_sort_predict(self, user_ids):
        original_uids = [self.recoxplainer_dataset.original_user_id.loc[uid]["user_id"] for uid in user_ids]

        batch_results = []
        for start in range(0, len(original_uids), self.batch_user_size):
            batch_uids = original_uids[start:start + self.batch_user_size]
            recbole_uids = self.recbole_dataset.token2id(self.recbole_dataset.uid_field, [str(u) for u in batch_uids])
            interactions = Interaction({"user_id": torch.tensor(recbole_uids, dtype=torch.int64)})

            try:
                scores = self.recbole_model.full_sort_predict(interactions.to(self.device))
            except NotImplementedError:
                item_tensor = self.recbole_dataset.get_item_feature().to(self.device)
                inter_len = len(interactions)
                new_inter = interactions.to(self.device).repeat_interleave(self.recbole_dataset.item_num)
                batch_size = len(new_inter)
                new_inter.update(item_tensor.repeat(inter_len))
                if batch_size <= self.batch_user_size:
                    scores = self.recbole_model.predict(new_inter)
                else:
                    scores = self._spilt_predict(new_inter, batch_size)

            batch_results.append(scores)

        all_scores = torch.cat(batch_results, dim=0)
        return all_scores.view(-1, self.recbole_dataset.item_num)

    def _spilt_predict(self, interaction, batch_size: int):
        spilt = {key: tensor.split(self.batch_size, dim=0) for key, tensor in interaction.interaction.items()}
        num_blocks = (batch_size + self.batch_size - 1) // self.batch_size
        results = []
        for i in range(num_blocks):
            current = {key: chunks[i] for key, chunks in spilt.items()}
            result = self.recbole_model.predict(Interaction(current).to(self.device))
            if len(result.shape) == 0:
                result = result.unsqueeze(0)
            results.append(result)
        return torch.cat(results, dim=0)

    def get_item2id_dict(self):
        return self.recbole_dataset.field2token_id[self.recbole_dataset.iid_field]


class _PatchedDataReader(DataReader):
    """DataReader subclass fixing a pandas 2.0 incompatibility: RecOXPlainer's DataReader
    used double-bracket indexing (`df[['col']]`) before calling `.nunique()`, relying on an
    implicit-int coercion pandas 2.0 removed. This overrides the property with single-bracket
    indexing, which returns a scalar directly."""

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = pd.read_csv(
                filepath_or_buffer=self.filepath_or_buffer, sep=self.sep,
                names=self.names, skiprows=self.skiprows, engine="python",
            )
            self._num_item = int(self._dataset["itemId"].nunique())
            self._num_user = int(self._dataset["userId"].nunique())
        return self._dataset

    @dataset.setter
    def dataset(self, new_data):
        self._dataset = new_data


def load_model(
    model_path: str, data_name: str, recoxplainer_config_path: str, full_sort: bool = False,
) -> Tuple[CustomModel, pd.DataFrame, DataReader]:
    """Loads a saved RecBole model checkpoint and generates recommendations."""
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # PyTorch >=2.6 changed torch.load's default weights_only to True, which breaks RecBole
    # checkpoints containing non-tensor objects. Patch it temporarily.
    _original_torch_load = torch.load
    torch.load = lambda *a, **kw: _original_torch_load(*a, weights_only=False, **kw)
    try:
        config, model, dataset, _, _, _ = load_data_and_model(model_file=model_path)
    finally:
        torch.load = _original_torch_load

    cfg = load_config(recoxplainer_config_path)
    data = _PatchedDataReader(**cfg[data_name])
    data.make_consecutive_ids_in_dataset(data.names)

    custom_model = CustomModel(model, dataset, data, config.final_config_dict["eval_batch_size"])
    rec = Recommender(data, custom_model)
    rec = rec.recommend_all(full_sort=full_sort, batch_size=10_000)
    return custom_model, rec, data


def post_hoc_ar(
    model: CustomModel, recommendations: pd.DataFrame, data: DataReader,
    min_support: float = 0.001, max_len: int = 2, metric: str = "lift",
    min_threshold: float = 0.001, min_confidence: float = 0.001, min_lift: float = 0.001,
) -> pd.DataFrame:
    """Generates post-hoc explanations using Association Rules."""
    explainer = ARPostHocExplainer(
        model, recommendations.copy(), data,
        min_support=min_support, max_len=max_len, metric=metric,
        min_threshold=min_threshold, min_confidence=min_confidence, min_lift=min_lift,
    )
    return explainer.explain_recommendations()


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
    rexbench's canonical ids, and renames to rexbench's schema (user_id/item_id/
    explanation_set) for metrics/explanation_quality.py."""
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