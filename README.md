# rexbench

A config-driven reproduction pipeline for an explainable-fairness RecSys study. One YAML
config plus one command regenerates every result; every experimental variable (seeds,
splits, hyperparameters, k, tail threshold) is declared in the config, not buried in code.
See [`REGISTRY.md`](REGISTRY.md) for every audited implementation this pipeline
deliberately does not run, with the reason. Findings referenced throughout as "AUDIT.md"
come from a prior reproducibility audit of the original (predecessor) pipeline — that
document lives outside this repository; the corrections it identified are applied here (see
below) and referenced by section for provenance.

This repository is self-contained: cloning it with its submodules and downloading the
datasets per `data/README.md` is everything needed to run it — no other local checkout is
required.

## Getting the code

```bash
git clone --recurse-submodules <this-repo-url> rexbench
cd rexbench
# if you cloned without --recurse-submodules:
git submodule update --init --recursive
```

## What this wraps, not reimplements

| Model family | Wraps | Adapter |
|---|---|---|
| Pop, BPR, MultiVAE, NeuMF, SLIM (Tier 2) | vendored RecBole (pip dependency) + `models/recbole_adapter.py`'s own `.inter` file writer | `models/recbole_adapter.py` |
| BPR/EBPR/UBPR/pUEBPR/UEBPR (Tier 1, required) — all five variants its engine dispatches on; the EBPR-codebase BPR is configured as `BPR_ebpr` to avoid colliding with RecBole's `BPR` | `external/ebpr/Code/{EBPR_model,engine_EBPR,data}.py` (submodule, your fork) | `models/ebpr_adapter.py` |
| PGPR (Tier 1, required) | `external/explanation-quality-recsys/models/PGPR/*.py` (submodule, your fork; the outer, actually-executed tree) | `models/pgpr_adapter.py` |
| AR, KNN explainers | `external/recoxplainer` (submodule, your fork) + `explainers/_bridge.py`'s bridging code (ported from the predecessor pipeline's explanation layer) | `explainers/recoxplainer_{ar,knn}.py` |

No method's training/scoring logic was reimplemented. Where an adapter had to call into
original code slightly differently than its own CLI script does (e.g. constructing a
`SampleGenerator`/`Config` directly instead of via `argparse`), that's I/O plumbing — see
each adapter module's docstring for exactly what's reused verbatim vs. adapted.

## Dependency layout

- **`external/`** — git submodules, each pinned to a specific commit on the correct branch of
  your own fork: `recoxplainer` (`master`, `ef17844`), `ebpr` (`reproducibility`, `a627034`,
  EBPR/UBPR/UEBPR/BPR), `explanation-quality-recsys` (`reproducibility`, `c250354`, PGPR). The
  branch is recorded in `.gitmodules` (`submodule.<name>.branch`) so `git submodule update
  --remote` tracks the right branch if you ever want to move the pin forward — run
  `git submodule status` to check what's currently checked out.
- **RecBole** — a normal pinned pip dependency (`recbole>=1.2` in `pyproject.toml`), the
  same as torch/numpy/pandas. Not vendored — it's a standard, unmodified open-source
  package with stable PyPI releases.
- **`kmeans-pytorch` and `ray[tune]`** — also plain pip dependencies, needed only because of
  how RecBole imports itself, not because rexbench uses either: `recbole.model.
  general_recommender`'s `__init__.py` eagerly imports all ~32 of its models (not just the
  5 rexbench actually configures), and one of them (`LDiffRec`) needs `kmeans_pytorch`;
  separately, `explainers/_bridge.py` imports `recbole.quick_start.quick_start` directly,
  which unconditionally imports `ray.tune`. Verified against RecBole 1.2.1's actual source
  that no other eagerly-loaded model needs an undeclared dependency (`NCL`/`faiss` and
  `NNCF`/`networkx`+`community` are real per-model needs but method-scoped, never triggered
  by importing the package) — so these two are the complete list, not a guess.
- **`python-box`** — `external/recoxplainer/recoxplainer/config.py` needs it (`from box
  import Box`), but that submodule's own `setup.py` declares no `install_requires` at all,
  so `pip install -e external/recoxplainer/` never pulls in any of its dependencies. Its
  `requirements.txt` is a full, stale notebook-environment snapshot (`torch==1.7.1`,
  `numpy==1.19.5`, `scikit-learn==0.24.1`, plus unrelated Jupyter/Dash packages) — do **not**
  `pip install -r` it, since it would downgrade this environment's pinned torch/numpy/
  scikit-learn below what RecBole/EBPR/PGPR need. Verified against the actual source that
  the code path `explainers/_bridge.py` touches (`config.py`, `data_reader/`, all four
  `explain/` classes, `recommender/`) needs nothing else beyond what's already declared.
- **The predecessor pipeline's small, stable surface** (6 dataset loaders, 2 split
  functions, RecBole `.inter`/YAML conversion, the RecOXPlainer explanation bridge) is
  absorbed directly into rexbench's own package (`core/dataset.py`, `core/splits.py`,
  `models/recbole_adapter.py`, `explainers/_bridge.py`) rather than kept as an external
  dependency — it's a small, already-audited surface (~670 lines) and rexbench is meant to
  supersede that pipeline's ad hoc scripts/notebooks, not depend on them.

## Environment

RecBole, EBPR, and PGPR are pinned to **mutually incompatible** dependency versions in their
original environments (torch 1.11–2.12, numpy 1.21–2.2 depending on which). Running all
three adapters in one process (required for `rexbench run` to be one command) needs a single
merged environment — confirmed feasible, with two small compatibility patches applied to the
`explanation-quality-recsys` fork (see below).

```bash
conda create -n rexbench python=3.10
conda activate rexbench
pip install -e .                          # installs the pinned set in pyproject.toml
pip install -e external/recoxplainer/     # the actively-patched fork, via the submodule
```

## Datasets

Not committed to this repository (several are 75MB–2.4GB, most carry their own license
terms). See [`data/README.md`](data/README.md) for exactly what to download, from where, and
where to place it under `data/raw/`.

## Running on Lightning AI

No local GPU needed — see [`LIGHTNING_AI.md`](LIGHTNING_AI.md) for step-by-step setup and
two things worth knowing before you start (EBPR's co-occurrence matrix is CPU-RAM-bound, not
GPU-bound; PGPR's knowledge-graph data still has no verified download source).

## Disclosed corrections to vendored source

Every run's `manifest.json` records these under `disclosed_corrections`. Applied directly to
the `explanation-quality-recsys` fork (`external/explanation-quality-recsys`, not a copy)
after verifying each against the actually-executed code path (`main.py`'s import graph —
`models/PGPR/*.py`, not the unused nested `models/PGPR/models/PGPR/` copy):

1. **`data_utils.py` rating/timestamp parsing** — both fields were gated on
   `dataset_name == "ml1m"`, hardcoding rating=0 and a nonsensical timestamp for every other
   dataset. Verified ml100k's review file has the identical 4-field layout as ml1m (both
   values genuinely present at the same positions) — a real bug, not a format difference for
   that pair. Fixed to parse both fields unconditionally for ml1m/ml100k; the committed fix
   (`external/explanation-quality-recsys` commit `c250354`) keeps a `dataset_name == "lastfm"`
   special case for rating/timestamp position, reflecting that lastfm's review file genuinely
   has a different layout — see that commit for the exact current logic, it was refined after
   my original proposal. (AUDIT.md section 3.1, approved before applying.)
2. **`test_agent.py` gender-fairness breakdown** — was commented-out dead code that still
   printed `nan`/`noOfUser=0`; re-enabled against variables already computed but unused.
   (AUDIT.md section 3.2, approved before applying.)
3. **`torch.load(...)` at `test_agent.py:191` and `train_transe_model.py:67`** —
   `weights_only=False` added explicitly (torch>=2.6 changed that default). An initial
   estimate of ~7 call sites needing this, plus an `np.float` usage, turned out on
   verification to only apply to files in the unused nested tree or an EBPR method never
   reached by this pipeline — see `core/determinism.py`'s `DISCLOSED_CORRECTIONS` for the
   full, corrected accounting.

**Status**: committed and pushed to your fork (`external/explanation-quality-recsys`,
branch `reproducibility`, commit `c250354`) — the submodule is pinned there. See
`old/CHANGELOG.md` for the diff of what this pipeline originally proposed.

**Also see REGISTRY.md's "Correctness finding" section** — two additional, more serious bugs
(a debug `break` truncating every EBPR training epoch to one batch; a hardcoded `10.999` stub
standing in for MAP@K) were found in the `ebpr` submodule while pinning it to the correct
branch, plus a related bug in this pipeline's own `EBPRModelAdapter` default — all three now
addressed (pin updated / adapter default fixed), documented there in full.

## Known scope limitations

- **RESOLVED: PGPR's train/test split now matches `DatasetBundle`.** `fit()` regenerates
  PGPR's `train.txt`/`test.txt` from `dataset.train`+`dataset.val` (merged — PGPR has no
  validation concept, and isn't wired into HPO) and `dataset.test`, translated to PGPR's own
  raw MovieLens ids. PGPR now shares the exact same test rows as every other model, and (as
  a side effect) also respects `SampleConfig` for smoke tests — see `REGISTRY.md` for the
  full writeup, including a real bug found and fixed along the way (`DATASET_DIR` was being
  pointed at the wrong on-disk location entirely).
- **PGPR runs on ML100K/ML1M only.** The other 6 datasets need a KG-construction step that
  doesn't exist yet — see `REGISTRY.md` for the per-dataset breakdown and rough cost estimate.
- **PGPR's knowledge-graph data has no verified canonical download source** — see
  `REGISTRY.md` and `data/README.md`. If you already have it from prior work,
  `rexbench stage-pgpr-kg --source <dir> --dataset ml100k` (or `ml1m`) copies it into
  rexbench's own `data/raw/pgpr_kg/<dataset>` with validation, so it travels with the rest of
  `data/raw/` when you sync to another machine.
- **BUGFIX**: `PGPRModelAdapter`'s precondition check used to count KG triples from
  `kg_final.txt` alone — verified against the vendored source that PGPR's own training code
  (`data_utils.py`/`knowledge_graph.py`) never actually reads that file, only
  `entities/`/`relations/`. `kg_final.txt`/`e_map.txt`/`r_map.txt` turn out to be leftover
  artifacts from an unrelated joint-kg/KGAT conversion, present for ml100k but not ml1m —
  the old check made ml1m look like it had zero KG triples and would have failed its
  precondition for no real reason. Fixed to count real triples from `relations/*.txt(.gz)`
  directly, falling back to the old `kg_final.txt` behavior only if no `relations/` dir
  exists at all.

## Fixed: RecBole was not using the exact rexbench split

Until this pass, `RecBoleModelAdapter` merged `dataset.train`+`dataset.val` into one file and
gave RecBole a `9:1` ratio to re-split internally (`eval_args.split.RS`) — RecBole then drew
its *own* fresh random split of that merged data with its own seeded RNG. That split was
consistent across RecBole models but was **not the literal `dataset.val` boundary**
EBPR/UBPR/UEBPR/PGPR train against via the `test` column mechanism — a real, previously
undiscovered violation of "the same split is reused by every model," which this whole
pipeline exists to guarantee. Fixed using RecBole's `benchmark_filename` mechanism (verified
against the vendored RecBole source, not guessed — `Dataset.build()` returns file-defined
split boundaries with no internal re-splitting when it's set): `models/recbole_adapter.py`
now writes `dataset.train`/`.val`/`.test` as three separate `.inter` files RecBole loads
verbatim. Also dropped the YAML-file round trip entirely in favor of building the RecBole
`Config` directly from a Python dict — simpler, not just the bug fix.

## Hyperparameter optimization

`ModelConfig.hpo` (and `DatasetOverride.hpo`, following the same per-dataset pattern as
`dataset_overrides` above) declares a config-driven grid or random search — see
`configs/hpo_example.yaml` for a worked example on EBPR. No external search library:
implemented directly in `core/hpo.py`/`core/hpo_search.py` using only the standard library —
Ray Tune and Optuna were both considered and rejected, since AUDIT.md already found Ray Tune
caused a hard version-mismatch crash in the ProtoMF baseline and this pipeline's merged
environment (numpy<2.0, torch>=2.0 exactly) is fragile enough already without adding another
pinned dependency.

Key properties:
- **Runs once per (dataset, model), not once per seed** — `core/runner.py`'s loop is now
  `dataset -> model -> seed`; HPO happens before the seed loop, and every seed then trains
  with the winning hyperparameters. Tuning per seed would multiply cost by
  `n_trials × n_seeds` for no benefit (the seed loop measures variance of the *chosen*
  config, not a reason to re-search it).
- **Evaluated against the validation split only** (`DatasetBundle.user_val_dict()`, new
  alongside the existing `user_test_dict()`) — never `test`, so tuning can't leak into the
  numbers actually reported.
- **Every trial is recorded**, ok or not, in a new `trials.parquet`/`.csv` output — same
  never-silently-drop philosophy as `results`/`failures`. A trial that crashes or hits
  `MetricRangeError` doesn't abort the search; it's logged and treated as worst-possible for
  selection purposes.
- Search types: `choice` (any values list), `int_uniform`/`uniform`/`loguniform` (numeric
  ranges) — `strategy: grid` requires every dimension to be `choice` (no natural grid over a
  continuous range; rejected at config-validation time otherwise).
- **Not wired in for PGPR** — it doesn't use `DatasetBundle`'s split at all (the scope
  limitation above), so HPO's val-based evaluation would inherit that same caveat; left out
  of `configs/hpo_example.yaml` for PGPR rather than silently pretending it's fixed.

## Persisted splits (reproducing the exact split across machines)

Without `split.store_dir` set, `DatasetBundle.train`/`.val`/`.test` are computed **in
memory, fresh, on every `rexbench run`** — reproducible only in the sense that the same raw
file + the same `split.random_state`/`sample.seed` deterministically re-derive the same rows
via `core/splits.py`. That's sufficient on one machine, but it means two different
environments (e.g. your laptop and a Lightning AI Studio) are trusting that pandas'/numpy's
sampling algorithms behave identically across versions to get byte-identical splits — usually
true, but not something this pipeline should have to assume silently for a reproducibility
study.

Setting `split.store_dir: data/splits/<name>` on a dataset removes that assumption entirely:
the first time `build_dataset_bundle()` runs for that dataset, it writes
`train.csv`/`val.csv`/`test.csv` (original id space) plus a `split_meta.json` fingerprint to
that directory; every subsequent call — on this machine or any other holding a copy of that
directory — loads those exact files instead of recomputing anything, and never touches the
raw dataset file again. `configs/tier1.yaml`/`full.yaml`/`hpo_example.yaml` all point at the
same `data/splits/<name>` (they share identical split settings, so the same materialized
split is correct for all three); `configs/smoke_test.yaml` uses `data/splits/smoke_test/<name>`
so its sampled split can never collide with a full-scale one at the same path.

If something *is* already persisted at `store_dir` but was computed under different
`split`/`sample` settings than the config now pointing at it, loading raises immediately
rather than silently reusing a mismatched split — see `core/dataset.py`'s
`_split_settings`/`_load_persisted_split`. `raw_path` is deliberately **not** part of that
check: it's recorded in `split_meta.json` for human debugging, but two configs pointing at
the same logical dataset via different paths (a laptop's dataset folder vs. a Studio's
`data/raw/...`) are expected and must not block reuse — that cross-machine case is the
entire reason this feature exists.

**Workflow** — split locally once, run the identical split anywhere:
```bash
# 1. On your machine, with the raw data already downloaded (data/README.md):
rexbench split --config configs/tier1.yaml
# -> writes data/splits/<dataset>/{train,val,test}.csv + split_meta.json for every dataset
#    that has store_dir set. Needs only pandas/pydantic -- no torch/RecBole/EBPR/PGPR -- so
#    this works even without the full merged environment installed.

# 2. Copy the result to wherever you're actually going to train (e.g. a Lightning AI Studio):
rsync -avz data/splits/ my-studio:rexbench/data/splits/

# 3. Run there -- rexbench run loads the persisted split instead of recomputing it, so the
#    exact same rows are used for every model regardless of which machine trained it:
rexbench run --config configs/tier1.yaml
```
This applies identically to `configs/smoke_test.yaml`'s sampled splits and to
`tier1.yaml`/`full.yaml`'s full-scale ones — sampling (`SampleConfig`) happens before the
split step either way, so the persisted files always reflect whatever `load_raw()` actually
produced.

`data/splits/` is gitignored, same as `data/raw/` — the persisted files still contain real
dataset rows (a reorganization of licensed content, not a reduction of it), so they travel
between machines by hand (`rsync`/`scp`/the Studio's file browser), never via git.

## Full-scale memory: LastFM1K and AMBAR need item-level aggregation to run at all

EBPR builds a **dense** item x item cosine-similarity matrix (`external/ebpr/Code/data.py`,
`create_explainability_matrix` / `create_neighborhood`), so its memory cost is quadratic in
catalogue size and lives in **CPU RAM, not GPU memory**. LastFM1K's raw item space is
*tracks*, and that makes a full-catalogue run impossible on any ordinary machine.

Measured directly from the raw 2.4GB file (19,150,868 plays, 992 users), not estimated:

| Entity | `min_item_interactions` | Items | Similarity matrix (float64) | Interactions kept |
|---|---|---|---|---|
| track | none | 961,417 | **7,395 GB** | 100% |
| track | 10 | 268,658 | 577 GB | 89.4% |
| track | 50 | 66,409 | 35 GB | 67.0% |
| track | 100 | 31,561 | 8.0 GB | 54.5% |
| artist | none | 107,398 | 92 GB | 100% |
| artist | 10 | 48,273 | 18.6 GB | 99.0% |
| **artist** | **20** | **35,474** | **10.1 GB** | **98.1%** |
| artist | 50 | 22,372 | 4.0 GB | 96.0% |

Two things fall out of that table:

- **Track level is not salvageable.** To fit in 32GB you have to filter down to ~100 plays
  per track, which throws away **45% of all interactions** — that is no longer the same
  dataset in any meaningful sense.
- **Artist level is nearly free.** Aggregating plays to artist collapses the catalogue 9x
  (961,417 -> 107,398) because the track catalogue is overwhelmingly a long tail of
  near-singletons, while the artist catalogue is dense. At `min_item_interactions: 20` it
  fits in 10GB while keeping 98% of the data.

So `configs/*.yaml` set LastFM1K to `loader: lastfm1k_artist` with
`filter: {min_item_interactions: 20}`. Verified end-to-end on the real file: **35,432 items,
18,138,905 interactions, 10.3 GB peak** (similarity + interaction matrix resident together).

This is also the more defensible choice scientifically, not just the cheaper one: the HetRec
LastFM-2K release is artist-based, and AMBAR — the other music dataset in this study — is
artist-based too, so the two are now directly comparable. **Report the filter threshold
alongside results**: k-core filtering is standard recommender-systems preprocessing, but it
is a real dataset-definition choice, not an implementation detail.

`FilterConfig` (`min_item_interactions` / `min_user_interactions`) is available for every
dataset and defaults to no filtering, so nothing else in the suite is affected. It applies
iterative k-core (dropping rare items can push users below their threshold and vice versa),
runs before the split and before `sample`, and is part of the persisted-split fingerprint —
a split materialized under one filter will not be silently reused under another.

### AMBAR has the same problem, with a cleaner fix

AMBAR's raw item space is 443,921 **tracks** (31,013 users, 3,311,462 ratings), so EBPR needs
443,921² × 8 = **1,577 GB** — plus another 110 GB for the dense users × items matrix.

Unlike LastFM1K, AMBAR ships an explicit `track_id -> artist_id` map in `tracks_info.csv`,
so `loader: ambar_artist` joins through it and reaches artist level **without discarding a
single interaction**: 30,667 items, all 3,311,462 ratings retained, **15.1 GB peak**
(measured). k-core filtering was the alternative and is strictly worse here — at
`min_item_interactions: 20` it reaches 11.7 GB but throws away 37% of the data.

This also puts AMBAR and LastFM1K both at artist level, so the two music datasets are
directly comparable — and it is what makes AMBAR's provider-fairness angle reachable later,
since `artists_info.csv` defines gender/country/continent per *artist*, not per track.

### Full-scale footprint, all eight datasets

Measured with the real configs (no sampling), peak = similarity + dense interaction matrix:

| Dataset | Users | Items | Interactions | Peak RAM | Status |
|---|---|---|---|---|---|
| ml100k | 943 | 1,682 | 100,000 | 0.02 GB | fine |
| ml1m | 6,040 | 3,706 | 1,000,209 | 0.3 GB | fine |
| coat | 290 | 300 | 11,600 | 0.001 GB | fine |
| electronics | 132,393 | 8,188 | 174,124 | 9.2 GB | fine |
| rentrunway | 105,508 | 5,850 | 192,462 | 5.2 GB | fine |
| lastfm1k (artist, ≥20) | 992 | 35,432 | 18,138,905 | 10.3 GB | fixed |
| ambar (artist) | 31,013 | 30,667 | 3,311,462 | 15.1 GB | fixed |
| amazon_digital_music | 100,952 | 70,519 | 130,434 | 96.7 GB | **unresolved** |

**`amazon_digital_music` is not a memory problem that filtering can fix.** It has 130,434
interactions spread over 100,952 users — 1.3 interactions per user, with 87.3% of users
having fewer than 2. Any filter that shrinks the catalogue enough to fit also destroys the
dataset: `min_item_interactions: 5` leaves 523 users and 1.1% of the interactions. This needs
a decision (a larger Amazon Digital Music dump, or excluding it from EBPR runs), not a
threshold.

**If you need to go smaller still**, `min_item_interactions: 50` gets artist level to 4 GB
while still keeping 96% of interactions. Changing EBPR's dense matrices to a chunked or
sparse top-N neighbourhood would remove the quadratic term entirely, but that means editing
vendored method code and is not done here.

## EBPR: best-model selection and `eval_every`

Two corrections and one new knob, all in the same area.

**The best model was never actually used.** `Engine.save_implicit` did
`best_model = self.model` — a *reference* to the live model, which training keeps mutating in
place. So `best_model` always ended up holding the final epoch's weights, and
`save_checkpoint(best_model, ...)` wrote the final epoch to disk under the best epoch's
filename, while `best_performance` reported the best epoch's metrics. The saved model and its
reported score disagreed. Fixed with `copy.deepcopy(self.model)` (the model is two embedding
tables, ~1 MB even for lastfm1k). `EBPRModelAdapter.fit()` now points the engine at that
snapshot, so `recommend()` scores with the best epoch's weights.

**Selection moved off the test split.** The per-epoch pass evaluated against `dataset.test` —
selecting the model on the very split rexbench then reports on, which inflates every number.
It now evaluates against `dataset.val`, so the test split never enters the training loop.
This also gives `dataset.val` a purpose; it was previously discarded, which is what left
items missing from the crosstab and produced the `KeyError` padding bug. `split_val` stays
`False` deliberately: with a `test` column present, `SampleGenerator._split_loo` honours it
verbatim, whereas `split_val=True` would make EBPR carve its own validation set out of train
and break the "same split for every model" contract.

**`eval_every` (new hyperparameter, default 1).** `Engine.evaluate()` is a full
leave-one-out pass with 100 sampled negatives per user and dominates EBPR's runtime. Since it
is what selects the best model it cannot simply be removed, but it can run less often:

| `eval_every` | passes over 50 epochs |
|---|---|
| 1 (default, original behaviour) | 50 |
| 5 (`configs/*` full-scale) | 11 |
| 10 | 6 |

The first and last epochs are always evaluated, so the fully-trained model is never excluded
from the candidates and `save_implicit`'s checkpoint write still fires. The trade is
granularity: a short-lived peak between two evaluated epochs can be missed. Note that
`dataset_overrides` replace the whole `hyperparameters` dict rather than merging it, so an
override that omits `eval_every` silently falls back to 1 — there is a test for that.

## Evaluation protocol: what every model shares

All reported metrics come from one path, identical for every algorithm:
`runner.py` calls `model.recommend(users, k)` and scores the result against
`dataset.user_test_dict()` through `metrics/dispatch.py`. No model computes its own reported
numbers — EBPR's internal `evaluate()` only picks which epoch's checkpoint to keep.

Three defects broke that premise and were fixed in this pass. All three change reported
values, so **results from before this fix are not comparable**.

**1. The evaluated population depended on the model.** `dataset_ndcg_k`/`map_at_k` skipped any
user whose list had fewer than `k` entries. Models that always return `k` items (EBPR,
RecBole) were scored on every user, while PGPR — whose candidates come from predicted paths
and can be short — had exactly those users dropped from its own average. Dropping them is not
neutral: the users a path-based model fails to reach are its hard cases. Demonstrated on a toy
frame: a model that found the relevant item for all 5 users but returned short lists for 3 of
them was averaged over 2. Now every user in the ground truth is scored; a short or empty list
simply scores lower. AUDIT.md had flagged this; it was preserved until now.

**2. NDCG was normalised by the hits found, not by the ideal.** The ideal ranking was
`sorted(r, reverse=True)` — the hits actually retrieved, pushed to the top — instead of
`min(|relevant|, k)` relevant items. That normalises recall away completely: a user with 20
relevant items who got 2 of them, both at ranks 1–2, scored **1.0** instead of 0.359
(measured). Any list whose hits sat at the top scored a perfect 1.0 regardless of how many
relevant items were missed, so the metric could not separate a model that finds everything
from one that finds almost nothing. `map_at_k` always normalised by `min(|relevant|, k)`, so
NDCG and MAP were not even consistent with each other. `gini` and `variance` are computed over
these per-user NDCG values and inherited the distortion.

**3. PGPR returned duplicate items.** It yields one entry per predicted *path*, and several
paths routinely end at the same item, so a top-5 could be `[42, 42, 42, 7, 7]` — two distinct
items filling five slots. Every other adapter returns `k` distinct items. Besides giving those
users fewer real options, it corrupted the item-side metrics, which count occurrences:
`item_coverage`, `entropy` and `item_exposure_gini` all treated one recommendation as several.
Now de-duplicated by item, keeping the highest-probability path (which also keeps that item's
explanation).

**Deliberately consistent, not a defect:** no adapter masks the user's training history out of
the candidate set, so all three may recommend already-seen items. That is the same rule for
every model. If you want held-out-only candidates, it has to change in all three at once.

**Still asymmetric, and not yet resolved:** PGPR trains on `train + val` (it has no validation
concept), while EBPR and RecBole train on `train` alone. PGPR therefore sees ~10% more
interactions than the others. Flag this in any head-to-head table until it is reconciled.

## Implemented metrics

Every metric below is selectable per-run from a config's `metrics:` block (`accuracy`,
`fairness`, `explanation`) and lands in `results.parquet`'s `metric` column under the name
in the first table column. Implementations live in `src/rexbench/metrics/`.

### Accuracy (`metrics.accuracy`) — on recommendation lists

| Name | What it measures | Source |
|---|---|---|
| `ndcg` | NDCG@k, averaged over users | Jarvelin & Kekalainen (2002), *Cumulated Gain-based Evaluation of IR Techniques*, ACM TOIS 20(4):422–446 — [doi:10.1145/582415.582418](https://doi.org/10.1145/582415.582418) |
| `map` | Mean Average Precision@k | Manning, Raghavan & Schutze (2008), *Introduction to Information Retrieval*, CUP, §8.4 — [online](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html) |

### Fairness (`metrics.fairness`) — on recommendation lists

**User-side** — how unequally recommendation *quality* is spread across users:

| Name | What it measures | Source |
|---|---|---|
| `gini` | Gini coefficient of the per-user NDCG@k distribution | Standard Gini coefficient; the predecessor pipeline cited no recsys source for this user-side use. Degenerate-case handling is rexbench's own fix (AUDIT.md §5) |
| `variance` | Mean squared pairwise difference of per-user NDCG@k | No external source — the predecessor pipeline's own O(n²) pairwise form, not textbook variance (preserved verbatim) |

**Item-side (provider-side)** — how *exposure* is spread across the catalog:

| Name | What it measures | Source |
|---|---|---|
| `entropy` | Shannon entropy of the item-appearance distribution across all top-k lists. Not normalized by log\|I\|, so not comparable across catalog sizes (preserved as-is) | Shannon (1948), *A Mathematical Theory of Communication*, BSTJ 27(3):379–423 — [doi:10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x); used as entropy-diversity for recommendations by Adomavicius & Kwon (2012) |
| `arp` | Average Recommendation Popularity — mean training popularity of recommended items. Reported in interaction-count units, not [0,1] | Abdollahpouri & Burke (2019), *Reducing Popularity Bias in Recommendation Over Time* — [arXiv:1906.11711](https://arxiv.org/abs/1906.11711) |
| `arp_normalized` | ARP min-max normalized to the dataset's own popularity range, so it's comparable across datasets | rexbench-specific (AUDIT.md §1.3) — no external source |
| `tail_coverage` | APLT — per-user share of the top-k list that is long-tail, averaged over users. Stored as `tail_coverage_rec` to disambiguate from the explanation metric of the same name | Abdollahpouri, Burke & Mobasher (2019), *Managing Popularity Bias in Recommender Systems with Personalized Re-ranking*, FLAIRS-32, pp. 413–418 — [arXiv:1901.07555](https://arxiv.org/abs/1901.07555) |
| `item_coverage` | Catalog coverage / aggregate diversity — fraction of the catalog appearing in *any* user's top-k | Adomavicius & Kwon (2012), *Improving Aggregate Recommendation Diversity Using Ranking-Based Techniques*, IEEE TKDE 24(5):896–911 — [doi:10.1109/TKDE.2011.15](https://doi.org/10.1109/TKDE.2011.15) |
| `item_exposure_gini` | Gini of per-item exposure over the full catalog (never-recommended items counted as zeros). The item-side counterpart to `gini` | Fleder & Hosanagar (2009), *Blockbuster Culture's Next Rise or Fall*, Management Science 55(5):697–712 — [doi:10.1287/mnsc.1080.0974](https://doi.org/10.1287/mnsc.1080.0974); also used by Adomavicius & Kwon (2012) |
| `tail_item_coverage` | Fraction of *distinct* long-tail items recommended to at least one user — catches the case where a good `tail_coverage` is really one tail item shown to everyone | Abdollahpouri et al. (2019) motivate this when introducing ACLT ([arXiv:1901.07555](https://arxiv.org/abs/1901.07555)); see the deviation note below |
| `tail_exposure_share` | Share of *position-discounted* exposure (1/log₂(1+rank)) going to long-tail items — penalizes burying tail items at the bottom of the list | Singh & Joachims (2018), *Fairness of Exposure in Rankings*, KDD '18 — [arXiv:1802.07281](https://arxiv.org/abs/1802.07281); Biega, Gummadi & Weikum (2018), *Equity of Attention*, SIGIR '18, pp. 405–414 — [doi:10.1145/3209978.3210063](https://doi.org/10.1145/3209978.3210063) |

Two notes worth knowing when citing these:

- **`tail_item_coverage` is not ACLT verbatim.** Abdollahpouri et al. state exactly this
  metric's motivation when introducing ACLT ("One problem with APLT is that it could be high
  even if all users get the same set of long tail items"), but their printed ACLT formula is
  an average *count* of tail items per list — which is APLT rescaled by list length and does
  not actually measure distinct coverage. This implements the stated intent, normalized to
  [0,1]. Cite it as tail-restricted catalog coverage, not as ACLT.
- **`tail_coverage` (APLT) was fixed, and its numbers changed.** The predecessor pipeline
  computed `tail_count / len(predictions_df)` over the entire prediction frame. That was
  wrong twice: it never truncated to top-k (so `tail_coverage@5` and `tail_coverage@10` came
  out identical in every run — the `k` column was meaningless for this metric), and it
  pooled all users into one ratio instead of averaging per-user percentages, silently
  weighting users with longer lists more heavily. Both are corrected to match APLT's
  published formula. **Results for this metric from runs predating the fix are not
  comparable** and should be regenerated.

A useful survey covering most of the item-side metrics above and their attributions:
Klimashevskaia, Jannach, Elahi & Trattner (2024), *A Survey on Popularity Bias in Recommender
Systems*, UMUAI — [doi:10.1007/s11257-024-09406-0](https://doi.org/10.1007/s11257-024-09406-0).

### Explanation quality (`metrics.explanation`) — on explanation sets

Computed only for (model, explainer) pairs that produced explanations (AR/KNN today).
**Citations pending** — these were ported verbatim from the predecessor pipeline, which
documented no sources, and the definitions have not yet been matched against the literature.
They are listed here for completeness, not as citable implementations of named metrics.

| Name | What it measures |
|---|---|
| `fidelity` | How often the explanation set contains the recommended item |
| `tail_coverage` | Share of explanation items that are long-tail (distinct from the recommendation-side metric of the same short name) |
| `diversity` | How varied explanation sets are across users |
| `explanation_arp` | Mean popularity of items inside explanation sets |
| `coverage` | Fraction of the catalog appearing in any explanation |
| `personalization` | 1 − mean pairwise Jaccard similarity between users' explanation sets |

## Running

```bash
rexbench split --config configs/stage1_tier1_local.yaml  # materialize data/splits/full/* from local raw data
rexbench run   --config configs/stage1_tier1.yaml       # STAGE 1: Tier 1 models, the 6 datasets verified to fit 32GB
rexbench run   --config configs/stage1_full.yaml        # STAGE 1: EVERYTHING (11 models x 6 datasets x 3 seeds)
rexbench stage-pgpr-kg --source <dir> --dataset ml100k  # copy PGPR's own KG data into data/raw/pgpr_kg/ml100k -- optional, only needed for PGPR
rexbench run --config configs/smoke_test.yaml  # everything we have, sampled tiny — run this first
rexbench run --config configs/tier1.yaml       # EBPR family + PGPR only
rexbench run --config configs/full.yaml        # + Tier 2 baselines + AR/KNN explainers
rexbench run --config configs/hpo_example.yaml # tier1.yaml + HPO search on EBPR
rexbench aggregate --run outputs/<run_id>
rexbench stats --run outputs/<run_id>
```

`DatasetConfig.sample` (`max_users`, `max_interactions_per_user`) is what makes
`smoke_test.yaml` small — pure pipeline-level data reduction in `core/dataset.py`, applied
before anything touches a model, so it never counts as changing a method's behavior.
Sampling by user (not by row) preserves each kept user's real interaction count, so a
dataset's sparsity character survives at the smaller scale — confirmed empirically:
`electronics`/`rentrunway` still show as clearly sparse after sampling, matching their
full-scale proportions. `max_interactions_per_user` exists specifically because
`max_users` alone doesn't bound the item catalog for datasets with huge per-user activity —
verified that 150 sampled `lastfm1k` users still touched 334,000 distinct items on their
own (~900GB for EBPR's item×item similarity matrix) without it.

Each run writes `outputs/<run_id>/`: `config.yaml` (copied verbatim), `manifest.json` (git
commit, library versions, hardware, disclosed corrections), `results.parquet`/`.csv` (long
format: dataset, model, explainer, metric, seed, k, value, status), `failures.parquet`/`.csv`
(one row per non-ok attempt, with exception type/message/traceback or precondition
measurements — never silently dropped), `trials.parquet`/`.csv` (one row per HPO trial, when
any model's config declares one — see Hyperparameter optimization above).

## Verification status

In the full merged environment (torch present) the whole suite runs:
```bash
PYTHONPATH=src python3 -m pytest tests/ -q          # 174 passed
```
Without torch installed, the adapter-level tests skip and the rest still run:
```bash
PYTHONPATH=src python3 -m pytest tests/test_config_schema.py tests/test_metrics_validate.py tests/test_dataset_bundle.py tests/test_hpo.py tests/test_significance.py tests/test_item_side_fairness.py tests/test_kcore_filter.py -v
```
103/103 passing there; 174/174 — covers the config schema (including all three real configs, the per-dataset
`dataset_overrides` mechanism, and `HPOConfig`/`HPOParamSpec` validation), the dataset/split/
tail-item logic (dependency-free after absorbing the predecessor pipeline's loaders —
`core/dataset.py` no longer needs RecBole just to import), the HPO grid/random sampling
logic (`core/hpo_search.py`, deliberately kept import-light the same way — `core/hpo.py`
itself needs the full model-adapter stack via `build_model`, so its pure sampling logic
lives in a separate module that doesn't), and, most importantly, the Gini fix (AUDIT.md
section 5) against
the exact degenerate inputs traced there: all-zero scores and a single-user dict now return
`NaN`, all-equal-nonzero scores return `0.0`, and the normal case is unchanged.

`test_item_side_fairness.py` covers the four item-side fairness metrics
(`item_coverage`, `item_exposure_gini`, `tail_item_coverage`, `tail_exposure_share`) against
hand-computed expected values rather than snapshots — including the discrete Gini maximum
`(n-1)/n` for fully concentrated exposure, the position-discount weights `1/log2(1+rank)`,
and the specific case `tail_item_coverage` exists to catch (one tail item shown to every
user scores 1/|tail|, not 1.0).

`test_significance.py` covers another real bug found this way: `rexbench stats` used to crash
(`scipy.stats.friedmanchisquare`: "Array shapes are incompatible for broadcasting") on any
metric where a model doesn't apply to every dataset — PGPR (ml100k/ml1m only) being the
obvious case. `friedman_test` was independently `dropna()`-ing each column, producing
differently-sized groups for a same-shape test; `quade_test`/`kendall_w` had no such
handling at all and would have silently returned `NaN` throughout instead of crashing. Fixed
with a shared `complete_cases()` (listwise deletion — restrict to datasets where every model
in the pivot has a value, the standard approach for an unbalanced repeated-measures design),
used consistently by all three, plus a `cmd_stats` check that skips a metric/k combination
that doesn't have at least 2 complete-case datasets after that restriction.

Also confirmed empirically (loading every dataset's real raw data with rexbench's own
loaders, no torch needed for this part): `amazon_digital_music` crashed on every load
(`pd.read_json`'s `convert_dates=True` default silently turned the `timestamp` column into
a datetime, breaking a later integer division) and `electronics`'s timestamp was a raw ISO
date string instead of a numeric epoch (harmless for EBPR, but would have broken
`RecBoleModelAdapter`'s `.inter` export). Both fixed in `core/dataset.py`.

## Per-dataset overrides (EBPR on sparse datasets)

`electronics`, `rentrunway`, and `amazon_digital_music` are far sparser than the other 5
datasets (79.6% / 68.0% / 87.3% of users have fewer than 2 interactions, measured directly
from the raw data) — too sparse for EBPR's default precondition
(`min_fraction_users_with_2plus: 0.5`), which exists because `create_explainability_matrix`/
`create_neighborhood`'s cosine-similarity item neighborhoods need real co-occurrence to mean
anything. `ModelConfig.dataset_overrides` (see `configs/tier1.yaml`/`full.yaml`'s EBPR entry)
lets a model declare a different `hyperparameters`/`precondition` block for specific
datasets, without duplicating the model under a second name — both configs now set a lower
sparsity floor (`0.05`) and a smaller item neighborhood (`5`, down from `20`) for those three
datasets, so EBPR/UBPR/UEBPR actually attempt training there instead of being skipped. The
expected outcome is a real but weak/degenerate explainability signal — the same class of
disclosed finding as the AR explainer's `model_fidelity=0.0` on these same three datasets —
not a crash.

`tests/test_precondition_checks.py` needs torch (`EBPRModelAdapter.recommend()` does real
tensor math) — it was not executable in the session this pipeline was built in (no merged
environment was available there; environment provisioning is left as a user action).
Every adapter was written against the real, directly-read source of each wrapped library (see
each module's docstring for exact function/line references) but full end-to-end training runs
were not executed in that session — treat the adapters as carefully-researched and internally
consistent, not as execution-verified, until run once in the built environment.