"""Adapter for the 5 Tier-2 RecBole models (Pop, BPR, MultiVAE, NeuMF, SLIM) plus ItemKNN
(registered but unused by any Tier config — AUDIT.md 1.1/3.9 found it dead in both
pipelines with no documented reason; see REGISTRY.md).

fit() converts the split into RecBole's .inter/.yaml format (ported from the predecessor
pipeline's preprocessing code, AUDIT.md found it correct) then runs RecBole's own
Config/create_dataset/data_preparation/Trainer.fit() call sequence UNCHANGED, including the
original SLIMElastic/Pop epoch overrides (AUDIT.md 4.6: a disclosed, pre-existing manual
convergence patch, not something rexbench added).

recommend() scores directly against the trained RecBole model/dataset objects (full_sort_
predict, with the identical NotImplementedError batched-fallback used by the explanation
bridge's CustomModel — see explainers/_bridge.py) rather than round-tripping through a saved
checkpoint + RecOXPlainer's DataReader — that round trip is only needed because RecOXPlainer's
Explainer requires its own ID space; plain top-k recommending does not. The AR/KNN
ExplainerAdapters (explainers/recoxplainer_*.py) reuse the checkpoint that RecBole's own
Trainer saves during fit() and do go through that bridge, unmodified.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import torch

import rexbench.vendor  # noqa: F401  (installs sys.path shims for the submodule imports below)
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed

from rexbench.config.schema import ModelConfig
from rexbench.core.dataset import DatasetBundle
from rexbench.models.base import ModelAdapter

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


def _create_recbole_yaml(
    output_path: str, data_path: str, data_name: str, checkpoint_dir: str,
    rating_field: Optional[str], timestamp_field: Optional[str], cols: str, ratio: str,
) -> None:
    fields = "USER_ID_FIELD: user_id\nITEM_ID_FIELD: item_id\n"
    if rating_field:
        fields += f"RATING_FIELD: {rating_field}\n"
    if timestamp_field:
        fields += f"TIME_FIELD: {timestamp_field}\n"

    template = (
        f"data_path: {data_path}\n"
        f"dataset: {data_name}\n"
        f"checkpoint_dir: {checkpoint_dir}\n"
        f"{fields}\n"
        f"load_col:\n"
        f"    inter: [{cols}]\n"
        f"\n"
        f"eval_args:\n"
        f"    split: {{'RS': [{ratio}]}}\n"
        f"    group_by: user\n"
        f"    order: RO\n"
        f"    mode: full\n"
    )
    Path(output_path).mkdir(parents=True, exist_ok=True)
    (Path(output_path) / f"{data_name}.yaml").write_text(template)


def _prepare_recbole_dataset(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    output_path: str, recbole_data_path: str, columns: list,
) -> None:
    """Writes RecBole .inter files (train+val merged, RecBole handles its own internal
    train/val split via the YAML RS ratio; test written separately) and the train/test YAML
    configs. Ported from the predecessor pipeline's preprocessing code (AUDIT.md 1.4 found
    it correct)."""
    out = Path(output_path)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    checkpoint_dir = str(Path(recbole_data_path) / "saved/")
    cols_yaml = ", ".join(c.split(":")[0] for c in columns)

    train_df, val_df, test_df = train_df.copy(), val_df.copy(), test_df.copy()
    train_df.columns = val_df.columns = test_df.columns = columns

    pd.concat([train_df, val_df]).to_csv(out / "train" / "train.inter", index=False, sep="\t")
    test_df.to_csv(out / "test" / "test.inter", index=False, sep="\t")

    _create_recbole_yaml(str(out), recbole_data_path, "train", checkpoint_dir, "rating", "timestamp", cols_yaml, "9, 1, 0")
    _create_recbole_yaml(str(out), recbole_data_path, "test", checkpoint_dir, "rating", "timestamp", cols_yaml, "10, 0, 10")


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

        _prepare_recbole_dataset(
            train_df=dataset.train[["user_id", "item_id", "rating", "timestamp"]],
            val_df=dataset.val[["user_id", "item_id", "rating", "timestamp"]],
            test_df=dataset.test[["user_id", "item_id", "rating", "timestamp"]],
            output_path=str(recbole_data_path),
            recbole_data_path=str(recbole_data_path),
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

        # Disclosed pre-existing patch (AUDIT.md 4.6), not introduced by rexbench.
        if self.recbole_model_name == "SLIMElastic":
            trainer.epochs = self.config.hyperparameters.get("epochs", 50)
        elif self.recbole_model_name == "Pop":
            trainer.epochs = self.config.hyperparameters.get("epochs", 1)

        trainer.fit(train_data, valid_data, saved=True, show_progress=False)

        self._model = model
        self._dataset = rb_dataset
        self._checkpoint_path = trainer.saved_model_file

    def _full_sort_scores(self, recbole_uids: list[int], device: str) -> torch.Tensor:
        """Mirrors the explanation bridge's CustomModel.full_sort_predict batching and
        NotImplementedError fallback (explainers/_bridge.py), in RecBole's own id space —
        no RecOXPlainer bridge needed for plain recommend()."""
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
        AR/KNN ExplainerAdapters load via explainers/_bridge.py's load_model()."""
        if self._checkpoint_path is None:
            raise RuntimeError(f"{self.name}: fit() has not been called yet")
        return self._checkpoint_path