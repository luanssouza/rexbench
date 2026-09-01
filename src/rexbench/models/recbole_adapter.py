"""Adapter for the 5 Tier-2 RecBole models (Pop, BPR, MultiVAE, NeuMF, SLIM) plus ItemKNN
(registered but unused by any Tier config — AUDIT.md 1.1/3.9 found it dead in both
pipelines with no documented reason; see REGISTRY.md).

fit() calls rexfair.rexfair.preprocessing.recbole.prepare_dataset (writes .inter/.yaml
exactly as rexfair's own pipeline does) then rexfair.rexfair.training.recbole.model_train
UNCHANGED — same Config/create_dataset/data_preparation/Trainer.fit() call sequence,
including the original SLIMElastic/Pop epoch overrides (AUDIT.md 4.6: a disclosed,
pre-existing manual convergence patch, not something rexbench added).

recommend() scores directly against the trained RecBole model/dataset objects (full_sort_
predict, with the identical NotImplementedError batched-fallback used by
rexfair.rexfair.explanation.model.CustomModel) rather than round-tripping through a saved
checkpoint + RecOXPlainer's DataReader — that round trip exists in rexfair only because
RecOXPlainer's Explainer needs its own ID space; plain top-k recommending does not need it.
The AR/KNN ExplainerAdapters (explainers/recoxplainer_*.py) reuse the checkpoint that
RecBole's own Trainer saves during fit() and do go through that bridge, unmodified.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

import rexbench.vendor  # noqa: F401
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed
from rexfair.preprocessing.recbole import prepare_dataset

from rexbench.config.schema import ModelConfig
from rexbench.core.dataset import DatasetBundle
from rexbench.models.base import ModelAdapter

# Same registry rexfair.rexfair.training.recbole.AVAILABLE_MODELS exposes (AUDIT.md 1.1).
from recbole.model.general_recommender import BPR
from recbole.model.general_recommender.itemknn import ItemKNN
from recbole.model.general_recommender.multivae import MultiVAE
from recbole.model.general_recommender.neumf import NeuMF
from recbole.model.general_recommender.pop import Pop
from recbole.model.general_recommender.slimelastic import SLIMElastic

AVAILABLE_MODELS = {
    "BPR": BPR, "ItemKNN": ItemKNN, "SLIMElastic": SLIMElastic,
    "MultiVAE": MultiVAE, "NeuMF": NeuMF, "Pop": Pop,
}

_RECBOLE_COLUMNS = ["user_id:token", "item_id:token", "rating:float", "timestamp:float"]


class RecBoleModelAdapter(ModelAdapter):
    family = "recbole"

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.name = config.name
        self.recbole_model_name = config.recbole_model
        self._model = None
        self._dataset = None
        self._checkpoint_path: str | None = None
        self._work_dir: tempfile.TemporaryDirectory | None = None

    def fit(self, dataset: DatasetBundle, seed: int, device: str) -> None:
        if self.recbole_model_name not in AVAILABLE_MODELS:
            raise KeyError(f"Unknown RecBole model {self.recbole_model_name!r}")

        self._work_dir = tempfile.TemporaryDirectory(prefix=f"rexbench-recbole-{dataset.name}-{self.name}-")
        work = Path(self._work_dir.name)
        recbole_data_path = work / "data"

        prepare_dataset(
            train_df=dataset.train[["user_id", "item_id", "rating", "timestamp"]],
            val_df=dataset.val[["user_id", "item_id", "rating", "timestamp"]],
            test_df=dataset.test[["user_id", "item_id", "rating", "timestamp"]],
            output_path=str(recbole_data_path),
            recbole_data_path=str(recbole_data_path),
            dataset_name="train",
            columns=_RECBOLE_COLUMNS,
        )

        config = Config(
            model=self.recbole_model_name,
            dataset="train",
            config_file_list=[str(recbole_data_path / "train.yaml")],
            config_dict={
                "seed": seed,
                "reproducibility": True,
                "device": device,
                **self.config.hyperparameters,
            },
        )
        init_seed(config["seed"], config["reproducibility"])
        init_logger(config)

        rb_dataset = create_dataset(config)
        train_data, valid_data, _ = data_preparation(config, rb_dataset)

        model = AVAILABLE_MODELS[self.recbole_model_name](config, train_data.dataset).to(config["device"])
        trainer = Trainer(config, model)

        # Identical to rexfair.rexfair.training.recbole.model_train — disclosed pre-existing
        # patch (AUDIT.md 4.6), not introduced by rexbench.
        if self.recbole_model_name == "SLIMElastic":
            trainer.epochs = self.config.hyperparameters.get("epochs", 50)
        elif self.recbole_model_name == "Pop":
            trainer.epochs = self.config.hyperparameters.get("epochs", 1)

        trainer.fit(train_data, valid_data, saved=True, show_progress=False)

        self._model = model
        self._dataset = rb_dataset
        self._checkpoint_path = trainer.saved_model_file

    def _full_sort_scores(self, recbole_uids: list[int], device: str) -> torch.Tensor:
        """Mirrors rexfair.rexfair.explanation.model.CustomModel.full_sort_predict's
        batching and NotImplementedError fallback, in RecBole's own id space (no
        RecOXPlainer bridge needed for plain recommend())."""
        interaction = Interaction({"user_id": torch.tensor(recbole_uids, dtype=torch.int64)}).to(device)
        try:
            return self._model.full_sort_predict(interaction)
        except NotImplementedError:
            item_tensor = self._dataset.get_item_feature().to(device)
            inter_len = len(interaction)
            new_inter = interaction.repeat_interleave(self._dataset.item_num)
            new_inter.update(item_tensor.repeat(inter_len))
            return self._model.predict(new_inter)

    def recommend(self, users: Sequence[int], k: int) -> pd.DataFrame:
        device = str(self._model.device)
        uid_field, iid_field = self._dataset.uid_field, self._dataset.iid_field
        recbole_uids = self._dataset.token2id(uid_field, [str(u) for u in users])

        scores = self._full_sort_scores(list(recbole_uids), device).view(len(users), -1)
        scores[:, 0] = float("-inf")  # RecBole reserves internal id 0 as a padding token

        topk_scores, topk_internal_ids = torch.topk(scores, k=min(k, scores.shape[1]), dim=1)
        topk_tokens = self._dataset.id2token(iid_field, topk_internal_ids.cpu().numpy())

        rows = []
        for row_i, user in enumerate(users):
            for rank, (token, score) in enumerate(
                zip(topk_tokens[row_i], topk_scores[row_i].tolist()), start=1
            ):
                rows.append({"user_id": user, "item_id": int(token), "rank": rank, "score": float(score)})
        return pd.DataFrame(rows, columns=["user_id", "item_id", "rank", "score"])

    def checkpoint_path(self) -> str:
        """Path to the .pth checkpoint RecBole's Trainer saved during fit() — the file the
        AR/KNN ExplainerAdapters load via rexfair.rexfair.explanation.explainers.load_model."""
        if self._checkpoint_path is None:
            raise RuntimeError(f"{self.name}: fit() has not been called yet")
        return self._checkpoint_path
