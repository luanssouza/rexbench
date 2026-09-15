"""Adapter for PGPR (baselines/explanation-quality-recsys/models/PGPR — the outer,
actually-executed tree; AUDIT.md's log evidence traces to these exact files, not the unused
nested `models/PGPR/models/PGPR/` copy).

RESOLVED (previously a disclosed scope limitation — see REGISTRY.md and git history for the
prior state): PGPR's own train/test files use the exact same raw MovieLens ids as
`u.data`/`ratings.dat` (verified against `myutils.py`'s `get_uid_to_kgid_mapping`/
`get_product_id_kgid_mapping`, which key their lookup tables by that raw id — the same value
already stored in `DatasetBundle.user_id_map`/`item_id_map`). `fit()` now regenerates PGPR's
`train.txt`/`test.txt` (+ `.gz`) from `dataset.train`+`dataset.val` (merged — PGPR has no
validation concept of its own and isn't wired into HPO, see README.md) and `dataset.test`,
translated back to raw ids, so PGPR shares the exact same test rows as every other model on
the dataset. Items with no KG entity node (a real, pre-existing gap — ml100k's KG covers
1424/1682 movies, ml1m 3265/3706) are silently skipped by PGPR's own existing
`generate_labels`/`load_reviews` logic (`if product_idx not in id2kgid: continue`) exactly as
they already were on PGPR's original split — this doesn't introduce that gap, just applies
the same graceful handling to rexbench's split instead.

Mechanically: `DATASET_DIR[name]` (from PGPR's own vendored `utils.py`) resolves via a
relative path (`'../../datasets/<name>'`) to `external/explanation-quality-recsys/datasets/
<name>` — a real, *git-tracked* directory inside the submodule holding a stale, incomplete
partial commit (missing `entities/`/`relations/`/`mappings/`/`train.txt.gz` entirely) — NOT
`models/PGPR/datasets/<name>` as an earlier version of this file assumed (a real bug: that
symlink was never actually read by anything). Since writing into a submodule's own tracked
files at runtime would be its own problem, `fit()` instead monkey-patches `DATASET_DIR[name]`
(a plain, mutable module-level dict) to point at a rexbench-owned, gitignored
`pgpr_runtime/<name>/` directory — see `_materialize_pgpr_dataset_dir` — which symlinks the
static KG structure from the staged `data/raw/pgpr_kg/<name>` (entities/relations/mappings,
unaffected by the split) and writes only train.txt/test.txt(.gz) fresh each fit() call.

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
import gzip
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
    """Real triple count for what PGPR's own training code actually reads. BUGFIX (found
    while staging ml1m's KG data): this used to count lines in `kg_final.txt` alone, but
    `data_utils.py`/`knowledge_graph.py` never read `kg_final.txt`/`e_map.txt`/`r_map.txt` at
    all -- PGPR loads `entities/*.txt.gz` and `relations/*.txt(.gz)` directly. Those three
    files are leftover artifacts from an unrelated joint-kg/KGAT conversion that happens to
    exist for ml100k but not ml1m, even though ml1m has perfectly good KG data of its own --
    the old check made ml1m look like it had zero triples and would have failed its
    precondition for no real reason. Each line in a relations/<relation>.txt file is one
    item's space-separated related-entity ids, so the real triple count for a relation is
    the token count across its lines, not the line count. Falls back to the old kg_final.txt
    line count only if there's no relations/ dir at all, rather than just returning 0."""
    relations_dir = kg_source_dir / "relations"
    if relations_dir.exists():
        by_relation: dict[str, Path] = {}
        for path in relations_dir.iterdir():
            if path.suffix not in (".txt", ".gz"):
                continue
            name = path.name.removesuffix(".gz").removesuffix(".txt")
            by_relation.setdefault(name, path)  # .txt and .txt.gz are duplicates; either is fine
        total = 0
        for path in by_relation.values():
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt") as f:
                total += sum(len(line.split()) for line in f)
        return total

    kg_final = kg_source_dir / "kg_final.txt"
    if kg_final.exists():
        with open(kg_final) as f:
            return sum(1 for _ in f)
    return 0


_PGPR_SPLIT_FILENAMES = ("train.txt", "train.txt.gz", "test.txt", "test.txt.gz")


def _write_pgpr_split_file(
    runtime_dir: Path, split_name: str, df: pd.DataFrame, user_id_map: dict, item_id_map: dict,
) -> None:
    """Writes `<split_name>.txt` and `.txt.gz` in PGPR's own review-file format (space
    separated `user item rating timestamp`, raw MovieLens ids — see module docstring),
    always overwriting whatever was there before: this directory is rexbench-owned working
    state, not the staged source, so there's no reuse-vs-recompute question here the way
    there is for data/splits/ -- every fit() call regenerates it fresh from the `dataset`
    object it was actually called with."""
    lines = [
        f"{user_id_map[int(row.user_id)]} {item_id_map[int(row.item_id)]} "
        f"{int(round(row.rating))} {int(row.timestamp)}"
        for row in df.itertuples(index=False)
    ]
    text = "\n".join(lines) + ("\n" if lines else "")
    (runtime_dir / f"{split_name}.txt").write_text(text)
    with gzip.open(runtime_dir / f"{split_name}.txt.gz", "wt") as f:
        f.write(text)


def _materialize_pgpr_dataset_dir(
    runtime_dir: Path, source_dir: Path, train_df: pd.DataFrame, test_df: pd.DataFrame,
    user_id_map: dict, item_id_map: dict,
) -> None:
    """Builds/refreshes a rexbench-owned PGPR dataset directory: symlinks everything static
    from the staged, read-only `data/raw/pgpr_kg/<name>` (entities/, relations/, mappings/,
    etc. — unaffected by which split is in use) but writes FRESH train.txt/test.txt(.gz)
    derived from rexbench's own DatasetBundle split. Never mutates data/raw/pgpr_kg itself
    (shared, rsync'd to other machines) and never writes into the vendored
    explanation-quality-recsys submodule's own datasets/<name> (git-tracked, and the
    directory PGPR's DATASET_DIR points at by default — see module docstring for why this
    function's caller overrides that instead)."""
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for entry in source_dir.iterdir():
        if entry.name in _PGPR_SPLIT_FILENAMES:
            continue
        link = runtime_dir / entry.name
        if not link.exists():
            link.symlink_to(entry)

    _write_pgpr_split_file(runtime_dir, "train", train_df, user_id_map, item_id_map)
    _write_pgpr_split_file(runtime_dir, "test", test_df, user_id_map, item_id_map)


class PGPRModelAdapter(ModelAdapter):
    family = "pgpr"

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.name = config.name
        self._dataset_name: str | None = None
        self._path_file: Path | None = None
        self._policy_file: Path | None = None

    def check_preconditions(self, dataset: DatasetBundle) -> PreconditionReport:
        min_triples = self.config.precondition_for(dataset.name).min_kg_triples or 1
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
        from utils import save_dataset, load_dataset, save_kg, get_logger
        from preprocess import generate_labels
        import train_transe_model
        import train_agent

        name = dataset.name
        if name not in DATASET_DIR:
            raise KeyError(f"PGPR has no dataset registry entry for {name!r}")
        self._dataset_name = name
        hp = self.config.hyperparameters_for(name)

        with _chdir(PGPR_ROOT):
            # The KG relation data isn't code (it doesn't live in the
            # explanation-quality-recsys submodule — see REGISTRY.md's "no verified
            # canonical download source" note), so it's placed manually per data/README.md
            # at data/raw/pgpr_kg/ (or via `rexbench stage-pgpr-kg`). See module docstring
            # for why this is materialized into pgpr_runtime/ (with DATASET_DIR
            # monkey-patched to point there) rather than symlinked directly into either
            # data/raw/pgpr_kg/ (shared, read-only, rsync'd to other machines) or the
            # submodule's own git-tracked datasets/<name> (stale/incomplete, and not ours to
            # write into).
            pgpr_kg_source = REPO_ROOT / "data" / "raw" / "pgpr_kg" / name
            if not pgpr_kg_source.exists():
                raise FileNotFoundError(
                    f"{pgpr_kg_source} not found — see data/README.md for how to obtain "
                    f"PGPR's knowledge-graph data before training on {name!r} (or run "
                    f"`rexbench stage-pgpr-kg --source <dir> --dataset {name}`)"
                )
            runtime_dir = REPO_ROOT / "pgpr_runtime" / name
            _materialize_pgpr_dataset_dir(
                runtime_dir, pgpr_kg_source,
                train_df=pd.concat([dataset.train, dataset.val], ignore_index=True),
                test_df=dataset.test,
                user_id_map=dataset.user_id_map, item_id_map=dataset.item_id_map,
            )
            DATASET_DIR[name] = str(runtime_dir)

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
            # BUGFIX (smoke test: every PGPR run crashed with "'NoneType' object has no
            # attribute 'info'"). train_transe_model/train_agent keep `logger = None` at
            # module level and only assign it inside their own main(), which this adapter
            # deliberately does not call -- it replicates main()'s body instead. Replicating
            # the logger setup too is the missing piece, not a behaviour change.
            train_transe_model.logger = get_logger(transe_args.log_dir + '/train_log.txt')
            train_transe_model.logger.info(str(transe_args))
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
            train_agent.logger = get_logger(agent_args.log_dir + '/train_log.txt')  # see above
            train_agent.logger.info(str(agent_args))
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
