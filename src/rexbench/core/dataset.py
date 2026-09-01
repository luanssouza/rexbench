"""Dataset loading, canonical ID remapping, and the once-per-dataset precomputation
(tail items, popularity, interaction stats) that every model/metric reuses.

Reuses rexfair's own loaders (rexfair.rexfair.data.loaders) and split functions
(rexfair.rexfair.preprocessing.splits) for every dataset that already has one —
AUDIT.md found those implementations correct. Two loaders (rentrunway, amazon_digital_music)
are new: AUDIT.md found no loader for either anywhere in the codebase (the "amazon" dataset's
raw source was untraceable at audit time; confirmed with the user to be Amazon Digital Music
going forward, see the plan's Context section).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

import rexbench.vendor  # noqa: F401  (installs sys.path shims before the rexfair import below)
from rexfair.data import loaders as rexfair_loaders
from rexfair.preprocessing.splits import random_split as _random_split
from rexfair.preprocessing.splits import temporal_split as _temporal_split

from rexbench.config.schema import DatasetConfig
from rexbench.metrics.tail import compute_tail_items

CANONICAL_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


def _load_rentrunway(path: str) -> pd.DataFrame:
    """New loader (AUDIT.md found none) for datasets/Clothes/renttherunway_final_data.json,
    one JSON object per line. `category`/`fit`/`size` fields exist for a future PGPR KG
    construction step (see REGISTRY.md) but aren't used by this rating-interaction loader."""
    df = pd.read_json(path, lines=True)
    df = df.dropna(subset=["user_id", "item_id", "rating"]).copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    ts = pd.to_datetime(df["review_date"], errors="coerce")
    df["timestamp"] = (ts.astype("int64") // 10**9).fillna(0).astype("int64")
    return df[CANONICAL_COLUMNS]


def _load_amazon_digital_music(path: str) -> pd.DataFrame:
    """New loader for datasets/Amazon/Digital_Music.jsonl — confirmed with the user as the
    true source of rexfair's "amazon" dataset (see plan Context)."""
    df = pd.read_json(path, lines=True)
    df = df.rename(columns={"asin": "item_id"})
    df["timestamp"] = (df["timestamp"] // 1000).astype("int64")  # ms -> s
    return df[CANONICAL_COLUMNS]


_LOADERS = {
    "ml100k": lambda raw: rexfair_loaders.load_ml100k(raw),
    "ml1m": lambda raw: rexfair_loaders.load_ml1m(raw),
    "lastfm1k": lambda raw: rexfair_loaders.load_lastfm1k(raw),
    "ambar": lambda raw: rexfair_loaders.load_ambar(raw),
    "coat": lambda raw: rexfair_loaders.load_coat(raw["train"], raw["test"]),
    "electronics": lambda raw: rexfair_loaders.load_electronics(raw),
    "rentrunway": _load_rentrunway,
    "amazon_digital_music": _load_amazon_digital_music,
}


def load_raw(config: DatasetConfig) -> pd.DataFrame:
    if config.loader not in _LOADERS:
        raise KeyError(f"no loader registered for {config.loader!r}; known: {sorted(_LOADERS)}")
    df = _LOADERS[config.loader](config.raw_path)
    if "rating" not in df.columns:
        df = df.assign(rating=1.0)
    if "timestamp" not in df.columns:
        df = df.assign(timestamp=0)
    return df[CANONICAL_COLUMNS].reset_index(drop=True)


@dataclass(frozen=True)
class InteractionStats:
    """AUDIT.md's minimum required precondition measurements, computed once per dataset and
    reused both as the EBPR-family precondition input and as manifest context for every run."""
    mean_interactions_per_user: float
    median_interactions_per_user: float
    fraction_users_lt_2: float
    num_users: int
    num_items: int
    num_interactions: int

    def as_dict(self) -> dict:
        return {
            "mean_interactions_per_user": self.mean_interactions_per_user,
            "median_interactions_per_user": self.median_interactions_per_user,
            "fraction_users_lt_2": self.fraction_users_lt_2,
            "num_users": self.num_users,
            "num_items": self.num_items,
            "num_interactions": self.num_interactions,
        }


def compute_interaction_stats(df: pd.DataFrame) -> InteractionStats:
    counts = df.groupby("user_id").size()
    return InteractionStats(
        mean_interactions_per_user=float(counts.mean()),
        median_interactions_per_user=float(counts.median()),
        fraction_users_lt_2=float((counts < 2).mean()),
        num_users=int(counts.shape[0]),
        num_items=int(df["item_id"].nunique()),
        num_interactions=int(len(df)),
    )


@dataclass
class DatasetBundle:
    name: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    num_users: int
    num_items: int
    tail_items: pd.Index
    popularity: pd.Series  # canonical item_id -> raw train-split interaction count
    user_id_map: dict  # canonical id -> original id
    item_id_map: dict
    kg_status: Literal["available", "needs_construction", "unsupported"]
    kg_path: Path | None
    interaction_stats: InteractionStats
    topk: list[int]
    tail_fraction: float

    def user_test_dict(self) -> dict:
        """{canonical_user_id: {canonical_item_id, ...}} from the held-out test split —
        the ground truth every accuracy/fairness metric evaluates recommendations against."""
        return {uid: set(g["item_id"]) for uid, g in self.test.groupby("user_id")}


def _remap_ids(df: pd.DataFrame, user_map: dict, item_map: dict) -> pd.DataFrame:
    out = df.copy()
    out["user_id"] = out["user_id"].map(user_map)
    out["item_id"] = out["item_id"].map(item_map)
    return out


def build_dataset_bundle(config: DatasetConfig) -> DatasetBundle:
    """Load raw data, split ONCE per dataset (deterministic given config.split.random_state,
    independent of model seed — AUDIT.md 4.2 found the split should be, and now
    contractually is, reused by every model trained on this dataset), remap ids to a
    canonical 0..n-1 space, and precompute tail items / popularity / interaction stats once."""
    raw = load_raw(config)
    stats = compute_interaction_stats(raw)

    if config.split.strategy == "random":
        train, val, test = _random_split(
            raw, val_size=config.split.val_size, test_size=config.split.test_size,
            random_state=config.split.random_state,
        )
    else:
        train, val, test = _temporal_split(
            raw, user_col="user_id", temp_col="timestamp",
            val_size=config.split.val_size, test_size=config.split.test_size,
            min_interactions=config.split.min_interactions,
        )

    users = sorted(raw["user_id"].unique())
    items = sorted(raw["item_id"].unique())
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {it: i for i, it in enumerate(items)}

    train = _remap_ids(train, user_map, item_map).reset_index(drop=True)
    val = _remap_ids(val, user_map, item_map).reset_index(drop=True)
    test = _remap_ids(test, user_map, item_map).reset_index(drop=True)

    tail_items = compute_tail_items(train, item_col="item_id", tail_fraction=config.tail_fraction)
    popularity = train.groupby("item_id").size().reindex(range(len(items)), fill_value=0)

    return DatasetBundle(
        name=config.name, train=train, val=val, test=test,
        num_users=len(users), num_items=len(items),
        tail_items=tail_items, popularity=popularity,
        user_id_map={v: k for k, v in user_map.items()},
        item_id_map={v: k for k, v in item_map.items()},
        kg_status=config.kg.status,
        kg_path=Path(config.kg.source_dir) if config.kg.source_dir else None,
        interaction_stats=stats, topk=list(config.topk), tail_fraction=config.tail_fraction,
    )
