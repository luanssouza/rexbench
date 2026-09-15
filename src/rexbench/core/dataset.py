"""Dataset loading, canonical ID remapping, and the once-per-dataset precomputation
(tail items, popularity, interaction stats) that every model/metric reuses.

Loaders for ml100k/ml1m/lastfm1k/ambar/coat/electronics are ported from the predecessor
pipeline's data-loading code (AUDIT.md found those implementations correct, reused
verbatim). Two loaders (rentrunway, amazon_digital_music) are new: AUDIT.md found no loader
for either anywhere in the codebase (the "amazon" dataset's raw source was untraceable at
audit time; confirmed with the user to be Amazon Digital Music going forward).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from rexbench.config.schema import DatasetConfig
from rexbench.core.splits import random_split as _random_split
from rexbench.core.splits import temporal_split as _temporal_split
from rexbench.metrics.tail import compute_tail_items

CANONICAL_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


def _load_ml100k(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = ["user_id", "item_id", "rating", "timestamp"]
    return df


def _load_ml1m(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="::", engine="python", header=None)
    df.columns = ["user_id", "item_id", "rating", "timestamp"]
    return df


def _load_lastfm1k(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", on_bad_lines="skip", header=None)
    df = df[[0, 4, 1]]
    df.columns = ["user_id", "item_id", "timestamp"]
    df.dropna(inplace=True)
    return df


def _load_lastfm1k_artist(path: str) -> pd.DataFrame:
    """LastFM1K aggregated to ARTIST level (column 2, `artid`) instead of track level
    (column 4, `traid`, which _load_lastfm1k above uses).

    This is a dataset-definition choice with a large practical consequence: the raw file has
    961,417 distinct tracks but only 107,398 distinct artists (measured, not estimated), and
    EBPR's item x item similarity matrix is dense — so track level needs 7.4 TB of RAM and
    artist level needs 92 GB, before any filtering. Combined with FilterConfig it becomes
    runnable on ordinary hardware; see README.md's "LastFM1K memory" section.

    Artist-level LastFM is also the more standard framing in the recommender-systems
    literature (the HetRec LastFM-2K release is artist-based), and it makes this dataset
    directly comparable to AMBAR, the other music dataset in this study, which is likewise
    artist-based.
    """
    df = pd.read_csv(path, sep="\t", on_bad_lines="skip", header=None)
    df = df[[0, 2, 1]]
    df.columns = ["user_id", "item_id", "timestamp"]
    df.dropna(inplace=True)
    return df


def _load_ambar(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = ["user_id", "item_id", "rating"]
    return df


def _coat_interactions(df: pd.DataFrame) -> pd.DataFrame:
    interactions = []
    for u, r in df.iterrows():
        ratings = r[0].split()
        for i, val in enumerate(ratings):
            if val != "0":
                interactions.append({"user_id": u, "item_id": i, "rating": int(val)})
    return pd.DataFrame(interactions)


def _load_coat(train_path: str, test_path: str) -> pd.DataFrame:
    train_df = _coat_interactions(pd.read_csv(train_path, header=None))
    test_df = _coat_interactions(pd.read_csv(test_path, header=None))
    return pd.concat([train_df, test_df]).sort_values("user_id").reset_index(drop=True)


def _load_electronics(path: str) -> pd.DataFrame:
    """Drops rows with missing user attributes — a MarketBias-specific debiasing filter.

    BUGFIX (found while investigating why EBPR appeared to only work on some datasets):
    the raw CSV's `timestamp` column is an ISO date string ("1999-06-13"), not a Unix
    epoch — the predecessor pipeline's loader passed it through unconverted. That's
    inconsistent with every other loader's CANONICAL_COLUMNS contract (timestamp = epoch
    seconds, int) and, more concretely, breaks RecBoleModelAdapter: `.inter` files declare
    `timestamp:float`, and RecBole fails to parse a literal date string as a float. EBPR
    itself isn't affected (its split path drops the timestamp column entirely), but fixing
    the loader here is the correct single source of truth rather than patching around it
    per-adapter.
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=["user_attr"])
    df["timestamp"] = (pd.to_datetime(df["timestamp"]).astype("int64") // 10**9).astype("int64")
    return df[["item_id", "user_id", "rating", "timestamp"]]


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
    true source of the predecessor pipeline's "amazon" dataset.

    BUGFIX: pd.read_json's default `convert_dates=True` auto-detects any column named
    "timestamp" (it matches the default date-like-column heuristic) and silently parses it
    as datetime64 instead of leaving the raw epoch-millisecond integers alone — so
    `df["timestamp"] // 1000` below raised `TypeError: cannot perform __floordiv__ with
    this index type: DatetimeArray` on every load, unconditionally. This was a real,
    100%-reproducible crash for this dataset, confirmed by loading the raw file directly.
    convert_dates=False keeps the column as the raw integers this function expects.
    """
    df = pd.read_json(path, lines=True, convert_dates=False)
    df = df.rename(columns={"asin": "item_id"})
    df["timestamp"] = (df["timestamp"] // 1000).astype("int64")  # ms -> s
    return df[CANONICAL_COLUMNS]


_LOADERS = {
    "ml100k": _load_ml100k,
    "ml1m": _load_ml1m,
    "lastfm1k": _load_lastfm1k,
    "lastfm1k_artist": _load_lastfm1k_artist,
    "ambar": _load_ambar,
    "coat": lambda raw: _load_coat(raw["train"], raw["test"]),
    "electronics": _load_electronics,
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
    df = df[CANONICAL_COLUMNS].reset_index(drop=True)
    # Filter BEFORE sample: filtering defines what the dataset is for this experiment,
    # sampling is a test-time reduction applied on top of that definition.
    df = _apply_kcore_filter(df, config.filter)
    if config.sample.max_users is not None:
        df = _subsample_users(df, config.sample.max_users, config.sample.seed)
    if config.sample.max_interactions_per_user is not None:
        df = _cap_interactions_per_user(df, config.sample.max_interactions_per_user, config.sample.seed)
    return df


def _apply_kcore_filter(df: pd.DataFrame, filter_config) -> pd.DataFrame:
    """Iterative k-core: repeatedly drop under-threshold items and users until neither
    changes. A single pass is not enough — dropping a rare item can push a user below the
    user threshold, which in turn can push further items below the item threshold.

    Stops early once stable, and hard-stops at `max_iterations` (a degenerate config can
    otherwise shrink the data one row at a time). Returns df unchanged when neither
    threshold is set, which is the default for every dataset.
    """
    min_item = filter_config.min_item_interactions
    min_user = filter_config.min_user_interactions
    if min_item is None and min_user is None:
        return df

    for _ in range(filter_config.max_iterations):
        before = len(df)
        if min_item is not None:
            item_counts = df["item_id"].value_counts()
            keep_items = item_counts[item_counts >= min_item].index
            df = df[df["item_id"].isin(keep_items)]
        if min_user is not None:
            user_counts = df["user_id"].value_counts()
            keep_users = user_counts[user_counts >= min_user].index
            df = df[df["user_id"].isin(keep_users)]
        if len(df) == before:
            break

    return df.reset_index(drop=True)


def _subsample_users(df: pd.DataFrame, max_users: int, seed: int) -> pd.DataFrame:
    """Keeps every interaction of a random `max_users` users, dropping the rest entirely —
    not a random row sample. Preserves each kept user's real interaction count, so EBPR's
    sparsity precondition behaves the same way on the sample as on the full dataset. A pure
    row sample would instead spread thin across every user, starving per-user density for
    reasons that have nothing to do with the real dataset. Does NOT by itself bound the item
    catalog for datasets where individual users are extremely active (see
    _cap_interactions_per_user) — verified empirically that 150 lastfm1k users still touch
    334,000 distinct items on their own."""
    users = df["user_id"].unique()
    if len(users) <= max_users:
        return df
    rng = np.random.default_rng(seed)
    chosen = rng.choice(users, size=max_users, replace=False)
    return df[df["user_id"].isin(chosen)].reset_index(drop=True)


def _cap_interactions_per_user(df: pd.DataFrame, max_interactions_per_user: int, seed: int) -> pd.DataFrame:
    """Keeps at most `max_interactions_per_user` random rows per user — the mechanism that
    actually bounds the item catalog for datasets like lastfm1k (huge per-user interaction
    counts), where _subsample_users alone doesn't: worst case the catalog is bounded by
    max_users * max_interactions_per_user, not by however many items a handful of very
    active users happen to have touched historically. Shuffle-then-head rather than a
    groupby().apply(lambda g: g.sample(...)) — the latter hits pandas' newer
    include_groups behavior (the grouping column can be dropped from the group passed to
    the lambda depending on pandas version), which is exactly the kind of version-fragility
    this pipeline elsewhere avoids by construction."""
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return shuffled.groupby("user_id", group_keys=False).head(max_interactions_per_user).reset_index(drop=True)


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

    def user_val_dict(self) -> dict:
        """Same shape as user_test_dict(), from the validation split — what core/hpo.py
        evaluates trials against. Never used for the numbers actually reported (those come
        from user_test_dict()), so tuning hyperparameters against this can't leak into the
        final results."""
        return {uid: set(g["item_id"]) for uid, g in self.val.groupby("user_id")}


def _remap_ids(df: pd.DataFrame, user_map: dict, item_map: dict) -> pd.DataFrame:
    out = df.copy()
    out["user_id"] = out["user_id"].map(user_map)
    out["item_id"] = out["item_id"].map(item_map)
    return out


SPLIT_META_FILENAME = "split_meta.json"


def _split_meta(config: DatasetConfig) -> dict:
    """Recorded alongside a materialized split for human debugging. `raw_path` is informational
    only -- see _split_settings for what's actually enforced on load."""
    return {
        "raw_path": config.raw_path,
        **_split_settings(config),
    }


def _split_settings(config: DatasetConfig) -> dict:
    """The subset of _split_meta that actually determines a persisted split's row content,
    given the same underlying dataset. Compared strictly against the stored meta on load so
    a persisted split silently reused from a differently configured run is a loud error, not
    a silent correctness bug.

    Deliberately excludes `raw_path`: the whole point of persisting a split is to reuse it on
    a *different machine*, where the same logical dataset legitimately lives at a different
    path (e.g. a laptop's `../datasets/ML100K/...` vs a Lightning AI Studio's
    `data/raw/ml100k/...`) -- without needing the raw file to be present there at all.
    Enforcing raw_path equality would defeat that exact use case."""
    return {
        "split": {
            "strategy": config.split.strategy,
            "val_size": config.split.val_size,
            "test_size": config.split.test_size,
            "random_state": config.split.random_state,
            "min_interactions": config.split.min_interactions,
        },
        "sample": {
            "max_users": config.sample.max_users,
            "max_interactions_per_user": config.sample.max_interactions_per_user,
            "seed": config.sample.seed,
        },
        "filter": {
            "min_item_interactions": config.filter.min_item_interactions,
            "min_user_interactions": config.filter.min_user_interactions,
            "max_iterations": config.filter.max_iterations,
        },
    }


def _persist_split(
    store_dir: Path, config: DatasetConfig, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    train[CANONICAL_COLUMNS].to_csv(store_dir / "train.csv", index=False)
    val[CANONICAL_COLUMNS].to_csv(store_dir / "val.csv", index=False)
    test[CANONICAL_COLUMNS].to_csv(store_dir / "test.csv", index=False)
    (store_dir / SPLIT_META_FILENAME).write_text(json.dumps(_split_meta(config), indent=2, default=str))


def _load_persisted_split(
    store_dir: Path, config: DatasetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Returns the persisted (train, val, test) in original raw-id space if `store_dir`
    already holds a complete, matching split; None if nothing is persisted yet (the caller
    should then compute and persist it). Raises if something IS persisted there but was
    computed from different settings -- reusing it anyway would silently break the
    "same split for every model" guarantee this pipeline exists to provide."""
    paths = {name: store_dir / f"{name}.csv" for name in ("train", "val", "test")}
    meta_path = store_dir / SPLIT_META_FILENAME
    if not (meta_path.exists() and all(p.exists() for p in paths.values())):
        return None

    stored_meta = json.loads(meta_path.read_text())
    stored_settings = {k: v for k, v in stored_meta.items() if k != "raw_path"}
    current_settings = _split_settings(config)
    if stored_settings != current_settings:
        raise ValueError(
            f"persisted split at {store_dir} was computed from different settings (split/sample) "
            f"than the dataset config using it now (stored: {stored_settings!r}, current: "
            f"{current_settings!r}). Refusing to silently reuse a mismatched split -- delete "
            f"{store_dir} to let it recompute, or point split.store_dir somewhere else."
        )
    return tuple(pd.read_csv(p) for p in paths.values())  # type: ignore[return-value]


def build_dataset_bundle(config: DatasetConfig) -> DatasetBundle:
    """Load raw data, split ONCE per dataset (deterministic given config.split.random_state,
    independent of model seed — AUDIT.md 4.2 found the split should be, and now
    contractually is, reused by every model trained on this dataset), remap ids to a
    canonical 0..n-1 space, and precompute tail items / popularity / interaction stats once.

    When config.split.store_dir is set, the split is materialized to disk the first time
    it's computed and loaded verbatim from there on every subsequent call (here or on a
    different machine holding a copy of that directory) instead of being recomputed --
    see _persist_split/_load_persisted_split."""
    store_dir = Path(config.split.store_dir) if config.split.store_dir else None
    persisted = _load_persisted_split(store_dir, config) if store_dir is not None else None

    if persisted is not None:
        train, val, test = persisted
        raw = pd.concat([train, val, test], ignore_index=True)
    else:
        raw = load_raw(config)
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
        if store_dir is not None:
            _persist_split(store_dir, config, train, val, test)

    stats = compute_interaction_stats(raw)
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