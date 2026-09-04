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
| Pop, BPR, MultiVAE, NeuMF, SLIM (Tier 2) | vendored RecBole (pip dependency) + `models/recbole_adapter.py`'s own `.inter`/YAML conversion (ported from the predecessor pipeline's preprocessing code) | `models/recbole_adapter.py` |
| BPR/EBPR/UBPR/UEBPR (Tier 1, required) | `external/ebpr/Code/{EBPR_model,engine_EBPR,data}.py` (submodule, your fork) | `models/ebpr_adapter.py` |
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

- **PGPR's train/test split is not rexbench's `DatasetBundle` split.** PGPR's own
  preprocessing pipeline has no parameterizable entry point for an arbitrary split — see
  `REGISTRY.md`. It's the one model not guaranteed to share exact train/test rows with every
  other model on the same dataset.
- **PGPR runs on ML100K/ML1M only.** The other 6 datasets need a KG-construction step that
  doesn't exist yet — see `REGISTRY.md` for the per-dataset breakdown and rough cost estimate.
- **PGPR's knowledge-graph data has no verified canonical download source** — see
  `REGISTRY.md` and `data/README.md`.

## Running

```bash
rexbench run --config configs/tier1.yaml   # EBPR family + PGPR only
rexbench run --config configs/full.yaml    # + Tier 2 baselines + AR/KNN explainers
rexbench aggregate --run outputs/<run_id>
rexbench stats --run outputs/<run_id>
```

Each run writes `outputs/<run_id>/`: `config.yaml` (copied verbatim), `manifest.json` (git
commit, library versions, hardware, disclosed corrections), `results.parquet`/`.csv` (long
format: dataset, model, explainer, metric, seed, k, value, status), `failures.parquet`/`.csv`
(one row per non-ok attempt, with exception type/message/traceback or precondition
measurements — never silently dropped).

## Verification status

Run without the merged environment installed:
```bash
PYTHONPATH=src python3 -m pytest tests/test_config_schema.py tests/test_metrics_validate.py tests/test_dataset_bundle.py -v
```
20/20 passing — covers the config schema (including both real configs, and the per-dataset
`dataset_overrides` mechanism below), the dataset/split/tail-item logic (now dependency-free
after absorbing the predecessor pipeline's loaders — `core/dataset.py` no longer needs
RecBole just to import), and, most importantly, the Gini fix (AUDIT.md section 5) against
the exact degenerate inputs traced there: all-zero scores and a single-user dict now return
`NaN`, all-equal-nonzero scores return `0.0`, and the normal case is unchanged.

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