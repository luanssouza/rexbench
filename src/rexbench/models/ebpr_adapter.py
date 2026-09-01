"""Adapter for BPR/EBPR/UBPR/UEBPR (baselines/EBPR).

fit() replicates baselines/EBPR/Code/train_EBPR.py's main() training loop call-for-call
(SampleGenerator -> create_explainability_matrix/popularity_vector/neighborhood ->
BPREngine -> per-epoch train_an_epoch/evaluate/save_implicit) — same functions, same order,
same variant dispatch via config['model']. The only deviation from train_EBPR.py: instead of
letting SampleGenerator draw its own random train/test split (its default, using
sklearn.train_test_split with config['test_rate']), we pass it a single DataFrame with an
explicit `test` column — a feature SampleGenerator.train_test_split_random already checks
for (baselines/EBPR/Code/data.py:207-218: `if 'test' in list(ratings): ...`) — so it uses
rexbench's DatasetBundle split instead of drawing its own. This is exactly the intended use
of that parameter, not a patch, and it's what makes "the same split reused by every model"
(AUDIT.md 4.2) hold for EBPR too.

recommend() is new: the original code only ever evaluates via LOO/random-split negative
sampling (Code/engine_EBPR.py's evaluate()), it has no top-k-for-arbitrary-users entry
point. BPR/EBPR/UBPR/UEBPR (baselines/EBPR/Code/EBPR_model.py:6) are all a plain embedding
dot product (`(embed_user * embed_item).sum(-1)`, EBPR_model.py:27-29) — recommend() applies
that exact trained scoring function to the full catalog per user, which is the only sensible
way to rank items from this architecture, not a new method.

check_preconditions() implements AUDIT.md's required EBPR precondition: SampleGenerator's
create_explainability_matrix/create_neighborhood build a dense users x items cosine-
similarity co-occurrence matrix (Code/data.py:322-360) that degrades silently on sparse
data rather than erroring — this precondition catches that case before fit() is attempted.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

import rexbench.vendor  # noqa: F401
from rexbench.vendor import EBPR_ROOT

from Code.data import SampleGenerator  # noqa: E402  (EBPR's own bare-import layout)
from Code.EBPR_model import BPREngine  # noqa: E402

from rexbench.config.schema import ModelConfig
from rexbench.core.dataset import DatasetBundle
from rexbench.models.base import ModelAdapter, PreconditionReport


@contextlib.contextmanager
def _chdir(path: Path):
    """EBPR's own code writes/reads relative paths (Output/checkpoints/, Output/results/) —
    it was designed to run with baselines/EBPR/ as the working directory, not given as an
    argument anywhere. This is a pure runtime adjustment (no source file touched)."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class EBPRModelAdapter(ModelAdapter):
    family = "ebpr"

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.name = config.name
        self.variant = config.variant  # "BPR" | "UBPR" | "EBPR" | "UEBPR"
        self._engine = None
        self._num_items = None

    def check_preconditions(self, dataset: DatasetBundle) -> PreconditionReport:
        required = self.config.precondition.min_fraction_users_with_2plus
        if required is None:
            return PreconditionReport.ok()
        stats = dataset.interaction_stats
        fraction_with_2plus = 1.0 - stats.fraction_users_lt_2
        satisfied = fraction_with_2plus >= required
        return PreconditionReport(
            satisfied=satisfied,
            measurements={
                "fraction_users_with_2plus_interactions": fraction_with_2plus,
                "mean_interactions_per_user": stats.mean_interactions_per_user,
                "median_interactions_per_user": stats.median_interactions_per_user,
            },
            requirement={"min_fraction_users_with_2plus": required},
            reason=(
                None if satisfied else
                f"{dataset.name}: only {fraction_with_2plus:.1%} of users have >=2 "
                f"interactions (need {required:.1%}) — the co-occurrence signal "
                f"create_explainability_matrix/create_neighborhood need "
                f"(Code/data.py:322-360) would be degenerate, not erroring, on this data"
            ),
        )

    def _build_config(self, dataset: DatasetBundle, seed: int, device: str) -> dict:
        hp = self.config.hyperparameters
        return {
            "model": self.variant,
            "dataset": dataset.name,
            "num_epoch": hp.get("num_epoch", 50),
            "batch_size": hp.get("batch_size", 512),
            "lr": hp.get("lr", 0.001),
            "sgd_momentum": hp.get("sgd_momentum", 0.9),
            "rmsprop_alpha": hp.get("rmsprop_alpha", 0.99),
            "rmsprop_momentum": hp.get("rmsprop_momentum", 0.0),
            "optimizer": hp.get("optimizer", "adam"),
            "num_users": dataset.num_users,
            "num_items": dataset.num_items,
            "test_rate": hp.get("test_rate", 0.2),
            "num_latent": hp.get("num_latent", 8),
            "weight_decay": hp.get("weight_decay", 0.0),
            "l2_regularization": hp.get("l2_regularization", 0.0),
            "use_cuda": device == "cuda",
            "device_id": 0,
            "top_k": max(dataset.topk),
            "loo_eval": hp.get("loo_eval", False),
            "neighborhood": hp.get("neighborhood", 20),
            "model_dir_explicit": "Output/checkpoints/{}_Epoch{}_MAP@{}_{:.4f}_NDCG@{}_{:.4f}_MEP@{}_{:.4f}_WMEP@{}_{:.4f}_Avg_Pop@{}_{:.4f}_EFD@{}_{:.4f}_Avg_Pair_Sim@{}_{:.4f}.model",
            "model_dir_implicit": "Output/checkpoints/{}_Epoch{}_NDCG@{}_{:.4f}_HR@{}_{:.4f}_MEP@{}_{:.4f}_WMEP@{}_{:.4f}_Avg_Pop@{}_{:.4f}_EFD@{}_{:.4f}_Avg_Pair_Sim@{}_{:.4f}.model",
        }

    def fit(self, dataset: DatasetBundle, seed: int, device: str) -> None:
        config = self._build_config(dataset, seed, device)
        self._num_items = dataset.num_items

        ratings = pd.concat(
            [dataset.train.assign(test=0), dataset.test.assign(test=1)], ignore_index=True
        ).rename(columns={"user_id": "userId", "item_id": "itemId"})[
            ["userId", "itemId", "rating", "timestamp", "test"]
        ]

        with _chdir(EBPR_ROOT):
            sample_generator = SampleGenerator(ratings, config, split_val=False)
            test_data = sample_generator.test_data_loader(config["batch_size"])
            _, _, _, explainability_matrix = sample_generator.create_explainability_matrix()
            _, _, _, test_explainability_matrix = sample_generator.create_explainability_matrix(include_test=True)
            popularity_vector = sample_generator.create_popularity_vector()
            test_popularity_vector = sample_generator.create_popularity_vector(include_test=True)
            neighborhood, _ = sample_generator.create_neighborhood()
            _, test_item_similarity_matrix = sample_generator.create_neighborhood(include_test=True)

            engine = BPREngine(config)
            best_performance = [0] * 8
            best_model = ""
            for epoch in range(config["num_epoch"]):
                train_loader = sample_generator.train_data_loader(config["batch_size"])
                engine.train_an_epoch(train_loader, explainability_matrix, popularity_vector, neighborhood, epoch_id=epoch)
                ndcg, hr, mep, wmep, avg_pop, efd, avg_pair_sim = engine.evaluate(
                    test_data, test_explainability_matrix, test_popularity_vector,
                    test_item_similarity_matrix, epoch_id=str(epoch) + " on test data",
                )
                best_model, best_performance = engine.save_implicit(
                    epoch, ndcg, hr, mep, wmep, avg_pop, efd, avg_pair_sim,
                    config["num_epoch"], best_model, best_performance, save_models=True,
                )

        self._engine = engine

    def recommend(self, users: Sequence[int], k: int) -> pd.DataFrame:
        model = self._engine.model
        device = next(model.parameters()).device
        user_idx = torch.tensor(list(users), dtype=torch.long, device=device)
        with torch.no_grad():
            user_latent = model.embed_user(user_idx)  # (n_users, d)
            item_latent = model.embed_item.weight  # (n_items, d) — same dot-product scoring
            scores = user_latent @ item_latent.T  # as EBPR_model.py:27-29's forward()

        topk_scores, topk_items = torch.topk(scores, k=min(k, scores.shape[1]), dim=1)
        rows = []
        for row_i, user in enumerate(users):
            for rank, (item, score) in enumerate(
                zip(topk_items[row_i].tolist(), topk_scores[row_i].tolist()), start=1
            ):
                rows.append({"user_id": user, "item_id": item, "rank": rank, "score": float(score)})
        return pd.DataFrame(rows, columns=["user_id", "item_id", "rank", "score"])
