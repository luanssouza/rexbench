"""Adapter for the 5 Tier-2 RecBole models (Pop, BPR, MultiVAE, NeuMF, SLIM) plus ItemKNN
(registered but unused by any Tier config — AUDIT.md 1.1/3.9 found it dead in both
pipelines with no documented reason; see REGISTRY.md).

fit() gives RecBole the exact train/val/test split from DatasetBundle via RecBole's
`benchmark_filename` mechanism (verified against the vendored RecBole source,
recbole/data/dataset/dataset.py: when `benchmark_filename` is set, Dataset.build() returns
those exact file-defined boundaries with no internal re-splitting/shuffling at all) rather
than merging train+val and letting RecBole draw its own fresh 90/10 split via `eval_args.
split.RS`. The RS-ratio approach used until now was a real bug, not a style choice: it meant
RecBole models were NOT training against the literal `dataset.val` boundary that EBPR/UBPR/
UEBPR/PGPR train against via the `test` column mechanism, silently breaking the "same split
reused by every model" guarantee this pipeline exists to provide. See the session that found
this for the full trace.

Config is now built entirely via `config_dict` (no YAML file round-trip) — simpler than the
previous file-writing code, not just the bug fix.

fit() otherwise runs RecBole's own Config/create_dataset/data_preparation/Trainer.fit() call
sequence UNCHANGED, including the original SLIMElastic/Pop epoch overrides (AUDIT.md 4.6: a
disclosed, pre-existing manual convergence patch, not something rexbench added).

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
from typing import Sequence

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

# Config._set_default_parameters (RecBole/recbole/config/configurator.py) special-cases this
# exact string to redirect data_path to RecBole's own bundled example dataset instead of the
# path we give it — avoided by construction (no rexbench dataset name contains a hyphen).
_FORBIDDEN_DATASET_TOKEN = "ml-100k"


def _prepare_recbole_benchmark_files(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    output_path: str, dataset_token: str, columns: list,
) -> Path:
    """Writes train/valid/test as separate RecBole .inter files using the `benchmark_filename`
    convention (`<dataset_token>.{train,valid,test}.inter`, no merging, no ratio) — the exact
    DatasetBundle split, verbatim. Returns the directory RecBole's `data_path` config value
    must resolve to one level above (Config appends `/<dataset>` itself)."""
    assert dataset_token != _FORBIDDEN_DATASET_TOKEN
    out = Path(output_path) / dataset_token
    out.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = train_df.copy(), val_df.copy(), test_df.copy()
    train_df.columns = val_df.columns = test_df.columns = columns

    train_df.to_csv(out / f"{dataset_token}.train.inter", index=False, sep="\t")
    val_df.to_csv(out / f"{dataset_token}.valid.inter", index=False, sep="\t")
    test_df.to_csv(out / f"{dataset_token}.test.inter", index=False, sep="\t")
    return out


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
        dataset_token = "train"  # fixed, arbitrary token — never the dataset's own name, so
                                  # it can never collide with _FORBIDDEN_DATASET_TOKEN

        dataset_dir = _prepare_recbole_benchmark_files(
            train_df=dataset.train[["user_id", "item_id", "rating", "timestamp"]],
            val_df=dataset.val[["user_id", "item_id", "rating", "timestamp"]],
            test_df=dataset.test[["user_id", "item_id", "rating", "timestamp"]],
            output_path=str(recbole_data_path),
            dataset_token=dataset_token,
            columns=_RECBOLE_COLUMNS,
        )

        hp = self.config.hyperparameters_for(dataset.name)
        config = Config(
            model=self.recbole_model_name,
            dataset=dataset_token,
            config_dict={
                "data_path": str(recbole_data_path),  # Config appends /<dataset_token> itself
                "checkpoint_dir": str(dataset_dir / "saved"),
                "benchmark_filename": ["train", "valid", "test"],
                "USER_ID_FIELD": "user_id",
                "ITEM_ID_FIELD": "item_id",
                "RATING_FIELD": "rating",
                "TIME_FIELD": "timestamp",
                "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
                "seed": seed,
                "reproducibility": True,
                "device": device,
                **hp,
            },
        )
        init_seed(config["seed"], config["reproducibility"])
        init_logger(config)

        rb_dataset = create_dataset(config)
        train_data, valid_data, _test_data = data_preparation(config, rb_dataset)

        model = AVAILABLE_MODELS[self.recbole_model_name](config, train_data.dataset).to(config["device"])
        trainer = Trainer(config, model)

        # Disclosed pre-existing patch (AUDIT.md 4.6), not introduced by rexbench.
        if self.recbole_model_name == "SLIMElastic":
            trainer.epochs = hp.get("epochs", 50)
        elif self.recbole_model_name == "Pop":
            trainer.epochs = hp.get("epochs", 1)

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
