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


def _leave_one_out(split: pd.DataFrame) -> pd.DataFrame:
    """One held-out row per user — the most recent by timestamp, ties broken by first
    occurrence. This is the shape Engine.evaluate() expects; see the note in fit()."""
    if split.empty:
        return split
    ordered = split.sort_values("timestamp", ascending=False, kind="mergesort")
    return ordered.groupby("user_id", sort=False, as_index=False).head(1).reset_index(drop=True)


class EBPRModelAdapter(ModelAdapter):
    family = "ebpr"

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.name = config.name
        # All five variants external/ebpr/Code/engine_EBPR.py dispatches on (its own
        # assert lists exactly these). The loss for each is selected by config["model"].
        self.variant = config.variant  # "BPR" | "UBPR" | "EBPR" | "pUEBPR" | "UEBPR"
        self._engine = None
        self._num_items = None
        self._best_epoch = None
        self._best_val_ndcg = None

    def check_preconditions(self, dataset: DatasetBundle) -> PreconditionReport:
        required = self.config.precondition_for(dataset.name).min_fraction_users_with_2plus
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
        hp = self.config.hyperparameters_for(dataset.name)
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
            # Default True, not False: with loo_eval=False, Engine.evaluate() (external/ebpr/
            # Code/engine_EBPR.py) returns (map, ndcg, mep, ...) via self._metron.cal_map_at_k(),
            # which is a stub in the vendored code (Code/metrics.py's cal_map_at_k always
            # `return 10.999`, real computation commented out below it) — confirmed present as
            # of external/ebpr commit a627034, not something this pipeline patched. fit() below
            # unpacks evaluate()'s return as (ndcg, hr, mep, ...) and calls save_implicit(),
            # which is the loo_eval=True contract; with loo_eval=False that call would silently
            # bind the stubbed map value to this adapter's `ndcg` variable and call the wrong
            # save_* method entirely (save_implicit checks best_performance[0], save_explicit
            # checks best_performance[1] — the two aren't interchangeable). loo_eval=True avoids
            # cal_map_at_k entirely (uses cal_hit_ratio_loo/cal_ndcg_loo instead) and matches
            # fit()'s actual evaluate()/save_implicit() call below.
            "loo_eval": hp.get("loo_eval", True),
            "neighborhood": hp.get("neighborhood", 20),
            # How often to run the per-epoch validation pass that selects the best model.
            # 1 = every epoch (the original behaviour, and the default so nothing changes
            # unless a config opts in). Higher values trade selection granularity for time:
            # Engine.evaluate() is a full leave-one-out pass with 100 sampled negatives per
            # user and dominates total runtime, so eval_every=5 over 50 epochs cuts it from
            # 50 passes to 11. The best model is then chosen among the epochs actually
            # evaluated -- it can miss a short-lived peak between checkpoints, which is the
            # whole trade. The final epoch is ALWAYS evaluated regardless (see fit()), so
            # the fully-trained model is never excluded from the candidates.
            "eval_every": max(1, int(hp.get("eval_every", 1))),
            "model_dir_explicit": "Output/checkpoints/{}_Epoch{}_MAP@{}_{:.4f}_NDCG@{}_{:.4f}_MEP@{}_{:.4f}_WMEP@{}_{:.4f}_Avg_Pop@{}_{:.4f}_EFD@{}_{:.4f}_Avg_Pair_Sim@{}_{:.4f}.model",
            "model_dir_implicit": "Output/checkpoints/{}_Epoch{}_NDCG@{}_{:.4f}_HR@{}_{:.4f}_MEP@{}_{:.4f}_WMEP@{}_{:.4f}_Avg_Pop@{}_{:.4f}_EFD@{}_{:.4f}_Avg_Pair_Sim@{}_{:.4f}.model",
        }

    def fit(self, dataset: DatasetBundle, seed: int, device: str) -> None:
        config = self._build_config(dataset, seed, device)
        self._num_items = dataset.num_items

        # EBPR's per-epoch evaluate()/save_implicit() pick the best epoch. Feeding
        # dataset.test here would select the model on the very split rexbench later reports
        # on -- test-set leakage that inflates every reported number. dataset.val is fed
        # instead, so EBPR's internal "test" set is our VALIDATION set and the real test
        # split never enters the training loop at all; rexbench scores it afterwards via
        # recommend(). This also stops dataset.val from being silently discarded, which is
        # what left items missing from the crosstab and caused the KeyError padding bug.
        #
        # The selection split is reduced to ONE held-out item per user by _leave_one_out
        # below, because Engine.evaluate() is a leave-one-out evaluator: it ranks each user's
        # single held-out item against 100 sampled negatives. Handing it rexbench's random
        # 10% split (11-17 items per user) broke it two ways. Semantically, each of those
        # items was ranked against the same negatives and counted separately, so the reported
        # HR/NDCG were not LOO quantities at all. Mechanically, metrics.py's
        # `pd.merge(full, test, on='user')` multiplies each user's rows by their own number
        # of held-out items: measured 12x for ml100k (92k -> 1.1M rows) and 19.5x for ml1m
        # (600k -> 11.7M), which is where ~80% of evaluate()'s runtime went (cProfile:
        # 367s of 459s inside the `subjects` setter, in groupby.rank and sort_values).
        #
        # Picking the most RECENT held-out interaction matches EBPR's own convention: its
        # _split_loo uses rank_latest == 1 when it derives a LOO split itself.
        #
        # This only affects which epoch's checkpoint is kept. The numbers rexbench reports
        # still come from recommend() scored against the FULL dataset.test, identically for
        # every model.
        #
        # split_val stays False on purpose: with a `test` column present, SampleGenerator's
        # _split_loo honours it verbatim (train = test==0, held-out = test==1) and does NOT
        # re-split internally. Passing split_val=True would make EBPR carve its own val out
        # of train, breaking the "same split for every model" contract -- the same class of
        # bug already fixed for RecBole.
        selection_split = _leave_one_out(dataset.val if len(dataset.val) else dataset.test)
        ratings = pd.concat(
            [dataset.train.assign(test=0), selection_split.assign(test=1)], ignore_index=True
        ).rename(columns={"user_id": "userId", "item_id": "itemId"})[
            ["userId", "itemId", "rating", "timestamp", "test"]
        ]

        # Code/data.py used to np.save() four matrices into Output/results/ on every call,
        # which both crashed on a fresh clone (the submodule only ships Output/checkpoints/)
        # and wrote ~587 GB across a full run. Those writes are now removed upstream in the
        # fork -- see core/determinism.py's DISCLOSED_CORRECTIONS. This mkdir is kept as
        # cheap insurance for any other relative path the vendored code may expect.
        (EBPR_ROOT / "Output" / "results").mkdir(parents=True, exist_ok=True)
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
            eval_every = config["eval_every"]
            last_epoch = config["num_epoch"] - 1
            for epoch in range(config["num_epoch"]):
                train_loader = sample_generator.train_data_loader(config["batch_size"])
                engine.train_an_epoch(train_loader, explainability_matrix, popularity_vector, neighborhood, epoch_id=epoch)
                # The last epoch is always evaluated: save_implicit only writes the
                # checkpoint on epoch == num_epoch - 1, and the fully-trained model must
                # stay a candidate for "best" no matter how eval_every falls.
                if epoch % eval_every and epoch != last_epoch:
                    continue
                ndcg, hr, mep, wmep, avg_pop, efd, avg_pair_sim = engine.evaluate(
                    test_data, test_explainability_matrix, test_popularity_vector,
                    test_item_similarity_matrix, epoch_id=str(epoch) + " on validation data",
                )
                best_model, best_performance = engine.save_implicit(
                    epoch, ndcg, hr, mep, wmep, avg_pop, efd, avg_pair_sim,
                    config["num_epoch"], best_model, best_performance, save_models=True,
                )

            # Use the BEST epoch's weights, not the last epoch's. save_implicit tracks the
            # best epoch by validation NDCG; before the engine_EBPR.py deepcopy fix it
            # handed back a reference to the live (final-epoch) model, so this swap would
            # have been a no-op. recommend() reads self._engine.model, so pointing the
            # engine at the snapshot is all that is needed.
            if isinstance(best_model, torch.nn.Module):
                engine.model = best_model
                self._best_epoch = best_performance[7]
                self._best_val_ndcg = best_performance[0]

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
