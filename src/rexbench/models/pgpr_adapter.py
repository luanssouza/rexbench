"""Adapter for PGPR (baselines/explanation-quality-recsys/models/PGPR — the outer,
actually-executed tree; AUDIT.md's log evidence traces to these exact files, not the unused
nested `models/PGPR/models/PGPR/` copy).

KNOWN, DISCLOSED SCOPE LIMITATION (see AUDIT.md's PGPR KG section and the plan's per-dataset
table): PGPR's entire pipeline (preprocess.py / train_transe_model.py / train_agent.py /
test_agent.py) is hardcoded around its own on-disk `datasets/<name>/train.txt` +
`test.txt` + pre-linked KG files, addressed by dataset-name string constants throughout
(myutils.py's DATASET_DIR/TMP_DIR/LABELS_DIR). There is no parameterizable entry point
for supplying an arbitrary train/test split. This adapter therefore runs PGPR's OWN
pre-existing ml100k/ml1m split (the same one AUDIT.md confirmed produced real results),
NOT rexbench's DatasetBundle split — unlike every other adapter, PGPR is not guaranteed to
share the exact same train/test rows as the other models on the same dataset. Reconciling
this (regenerating PGPR's train.txt/test.txt from DatasetBundle, keyed to PGPR's own
review-uid <-> KG-uid mapping) is real, scoped follow-up work — see REGISTRY.md.

fit() replicates preprocess.py's main() body (same function calls, same order — AmazonDataset
-> KnowledgeGraph -> generate_labels), then calls train_transe_model.train(args) and
train_agent.train(args) unmodified. recommend()/explain() call test_agent.predict_paths
unmodified and read back the paths it pickles: each predicted path already ends in the
recommended item, and the path itself IS this model's native explanation (see
models/base.py's ModelAdapter.explain docstring for why PGPR implements explain() while
RecBole/EBPR adapters don't).

check_preconditions() gates on dataset.kg_status == "available" (AUDIT.md's per-dataset KG
table) and, when available, counts real KG triples against the requirement in config.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import pickle
from pathlib import Path
from typing import Sequence

import torch
import pandas as pd

import rexbench.vendor  # noqa: F401
from rexbench.vendor import PGPR_ROOT, REPO_ROOT

from rexbench.config.schema import ModelConfig
from rexbench.core.dataset import DatasetBundle
from rexbench.models.base import Explanation, ModelAdapter, PreconditionReport


@contextlib.contextmanager
def _chdir(path: Path):
    """PGPR's DATASET_DIR/TMP_DIR/LABELS_DIR entries are all relative — this pipeline was
    designed to run with models/PGPR/ as the working directory."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _count_kg_triples(kg_source_dir: Path) -> int:
    kg_final = kg_source_dir / "kg_final.txt"
    if not kg_final.exists():
        return 0
    with open(kg_final) as f:
        return sum(1 for _ in f)


class PGPRModelAdapter(ModelAdapter):
    family = "pgpr"

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.name = config.name
        self._dataset_name: str | None = None
        self._path_file: Path | None = None
        self._policy_file: Path | None = None

    def check_preconditions(self, dataset: DatasetBundle) -> PreconditionReport:
        min_triples = self.config.precondition.min_kg_triples or 1
        if dataset.kg_status != "available":
            return PreconditionReport(
                satisfied=False,
                measurements={"kg_status": dataset.kg_status},
                requirement={"kg_status": "available"},
                reason=(
                    f"{dataset.name}: KG status is {dataset.kg_status!r}, not 'available' "
                    f"(AUDIT.md's PGPR KG table) — PGPR requires a pre-linked knowledge "
                    f"graph, which this dataset does not currently have"
                ),
            )
        n_triples = _count_kg_triples(dataset.kg_path) if dataset.kg_path else 0
        satisfied = n_triples >= min_triples
        return PreconditionReport(
            satisfied=satisfied,
            measurements={"kg_triples": n_triples},
            requirement={"min_kg_triples": min_triples},
            reason=None if satisfied else f"{dataset.name}: only {n_triples} KG triples found (need >= {min_triples})",
        )

    def fit(self, dataset: DatasetBundle, seed: int, device: str) -> None:
        from utils import DATASET_DIR, TMP_DIR  # PGPR's own module, on sys.path via vendor
        from data_utils import AmazonDataset
        from knowledge_graph import KnowledgeGraph
        from utils import save_dataset, load_dataset, save_kg
        from preprocess import generate_labels
        import train_transe_model
        import train_agent

        name = dataset.name
        if name not in DATASET_DIR:
            raise KeyError(f"PGPR has no dataset registry entry for {name!r}")
        self._dataset_name = name
        hp = self.config.hyperparameters

        with _chdir(PGPR_ROOT):
            # train_ml100k.sh (the script that produced AUDIT.md's confirmed successful
            # PGPR runs) does `cd models/PGPR` before running these scripts, so
            # DATASET_DIR['./datasets/<name>'] resolves relative to models/PGPR/. The KG
            # relation data isn't code (it doesn't live in the explanation-quality-recsys
            # submodule — see REGISTRY.md's "no verified canonical download source" note),
            # so it's fetched separately by `rexbench data fetch` into data/raw/pgpr_kg/ and
            # symlinked in here; no submodule source file touched.
            local_datasets = Path("datasets")
            if not local_datasets.exists():
                pgpr_kg_dir = REPO_ROOT / "data" / "raw" / "pgpr_kg"
                if not pgpr_kg_dir.exists():
                    raise FileNotFoundError(
                        f"{pgpr_kg_dir} not found — run `rexbench data fetch --dataset pgpr_kg` "
                        f"(or see data/README.md) before training PGPR on {name!r}"
                    )
                local_datasets.symlink_to(pgpr_kg_dir)

            tmp_dir = Path(TMP_DIR[name])
            tmp_dir.mkdir(parents=True, exist_ok=True)

            pgpr_dataset = AmazonDataset(DATASET_DIR[name])
            save_dataset(name, pgpr_dataset)
            pgpr_dataset = load_dataset(name)
            kg = KnowledgeGraph(pgpr_dataset)
            kg.compute_degrees()
            save_kg(name, kg)
            generate_labels(name, "train")
            generate_labels(name, "test")

            transe_args = argparse.Namespace(
                dataset=name, name="transe", seed=seed, device=device,
                epochs=hp.get("transe_epochs", 30), batch_size=hp.get("transe_batch_size", 64),
                lr=hp.get("transe_lr", 0.5), weight_decay=hp.get("transe_weight_decay", 0),
                l2_lambda=hp.get("l2_lambda", 0), max_grad_norm=hp.get("max_grad_norm", 5.0),
                embed_size=hp.get("embed_size", 300), num_neg_samples=hp.get("num_neg_samples", 5),
                steps_per_checkpoint=hp.get("steps_per_checkpoint", 200),
                log_dir=str(tmp_dir / "transe"),
            )
            Path(transe_args.log_dir).mkdir(parents=True, exist_ok=True)
            train_transe_model.train(transe_args)
            train_transe_model.extract_embeddings(transe_args)

            agent_args = argparse.Namespace(
                dataset=name, name="agent", seed=seed, device=device,
                epochs=hp.get("agent_epochs", 50), batch_size=hp.get("agent_batch_size", 32),
                lr=hp.get("agent_lr", 1e-4), max_acts=hp.get("max_acts", 250),
                max_path_len=hp.get("max_path_len", 3), gamma=hp.get("gamma", 0.99),
                ent_weight=hp.get("ent_weight", 1e-3), act_dropout=hp.get("act_dropout", 0.5),
                state_history=hp.get("state_history", 1), hidden=hp.get("hidden", [512, 256]),
                log_dir=str(tmp_dir / "agent"),
            )
            Path(agent_args.log_dir).mkdir(parents=True, exist_ok=True)
            train_agent.train(agent_args)

            self._policy_file = Path(agent_args.log_dir) / f"policy_model_epoch_{agent_args.epochs}.ckpt"
            self._agent_args = agent_args

    def _predict_paths(self, k: int) -> dict:
        import test_agent

        with _chdir(PGPR_ROOT):
            path_file = Path(self._agent_args.log_dir) / "paths.pkl"
            predict_args = argparse.Namespace(
                **vars(self._agent_args), topk=[max(25, k), max(2 * k, 10), 1],
            )
            test_agent.predict_paths(str(self._policy_file), str(path_file), predict_args)
            with open(path_file, "rb") as f:
                return pickle.load(f)

    def recommend(self, users: Sequence[int], k: int) -> pd.DataFrame:
        predicts = self._predict_paths(k)
        by_user: dict[int, list[tuple[float, int, list]]] = {}
        for path, prob in zip(predicts["paths"], predicts["probs"]):
            uid = int(path[0][-1])
            item_id = int(path[-1][-1][-1])
            by_user.setdefault(uid, []).append((prob, item_id, path))

        rows = []
        for uid in users:
            ranked = sorted(by_user.get(uid, []), key=lambda t: -t[0])[:k]
            for rank, (prob, item_id, _path) in enumerate(ranked, start=1):
                rows.append({"user_id": uid, "item_id": item_id, "rank": rank, "score": float(prob)})
        return pd.DataFrame(rows, columns=["user_id", "item_id", "rank", "score"])

    def explain(self, user_id: int, item_id: int):
        predicts = self._predict_paths(10)
        for path, prob in zip(predicts["paths"], predicts["probs"]):
            if int(path[0][-1]) == user_id and int(path[-1][-1][-1]) == item_id:
                return Explanation(kind="pgpr_path", payload={"path": path, "probability": float(prob)})
        return None
