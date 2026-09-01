# rexbench

A config-driven reproduction pipeline for the explainable-fairness RecSys study audited in
[`/audit/AUDIT.md`](../audit/AUDIT.md). One YAML config plus one command regenerates every
result; every experimental variable (seeds, splits, hyperparameters, k, tail threshold) is
declared in the config, not buried in code. See [`REGISTRY.md`](REGISTRY.md) for every
audited implementation this pipeline deliberately does not run, with the reason.

## What this wraps, not reimplements

| Model family | Wraps | Adapter |
|---|---|---|
| Pop, BPR, MultiVAE, NeuMF, SLIM (Tier 2) | `rexfair/rexfair/training/recbole.py` + vendored RecBole | `models/recbole_adapter.py` |
| BPR/EBPR/UBPR/UEBPR (Tier 1, required) | `baselines/EBPR/Code/{EBPR_model,engine_EBPR,data}.py` | `models/ebpr_adapter.py` |
| PGPR (Tier 1, required) | `baselines/explanation-quality-recsys/models/PGPR/*.py` (the outer, actually-executed tree) | `models/pgpr_adapter.py` |
| AR, KNN explainers | `rexfair/rexfair/explanation/explainers.py` + `recoxplainer.explain` | `explainers/recoxplainer_{ar,knn}.py` |

No method's training/scoring logic was reimplemented. Where an adapter had to call into
original code slightly differently than its own CLI script does (e.g. constructing a
`SampleGenerator`/`Config` directly instead of via `argparse`), that's I/O plumbing — see
each adapter module's docstring for exactly what's reused verbatim vs. adapted.

## Environment

RecBole, EBPR, and PGPR are pinned to **mutually incompatible** dependency versions in their
own separate conda environments already on this machine (`recbole`/`mexfair`: torch 2.12,
numpy 1.26; `ebpr`: torch 2.7, numpy 2.2; `eqr`: torch 1.11, numpy 1.21). Running all three
adapters in one process (required for `rexbench run` to be one command) needed a single
merged environment — confirmed feasible, with two small compatibility patches to vendored
source (see below).

```bash
conda create -n rexbench python=3.10
conda activate rexbench
pip install -e rexbench/                 # installs the pinned set in rexbench/pyproject.toml
pip install -e recoxplainer/              # the actively-patched fork — NOT baselines/recoxplainer
```

## Disclosed corrections to vendored source

Every run's `manifest.json` records these under `disclosed_corrections`. Applied directly to
the vendored files (not copies) after verifying each against the actually-executed code path
(`main.py`'s import graph — `baselines/explanation-quality-recsys/models/PGPR/*.py`, not the
unused nested `models/PGPR/models/PGPR/` copy):

1. **`data_utils.py` rating/timestamp parsing** — both fields were gated on
   `dataset_name == "ml1m"`, hardcoding rating=0 and a nonsensical timestamp for every other
   dataset. Verified ml100k's review file has the identical 4-field layout as ml1m (both
   values genuinely present at the same positions) — a real bug, not a format difference.
   Fixed to parse both fields unconditionally. (AUDIT.md 3.1, approved before applying.)
2. **`test_agent.py` gender-fairness breakdown** — was commented-out dead code that still
   printed `nan`/`noOfUser=0`; re-enabled against variables already computed but unused.
   (AUDIT.md 3.2, approved before applying.)
3. **`torch.load(...)` at `test_agent.py:191` and `train_transe_model.py:67`** —
   `weights_only=False` added explicitly (torch>=2.6 changed that default; the identical
   pattern already exists in `rexfair/rexfair/explanation/explainers.py`). An initial
   estimate of ~7 call sites needing this, plus an `np.float` usage, turned out on
   verification to only apply to files in the unused nested tree or an EBPR method never
   reached by this pipeline — see `core/determinism.py`'s `DISCLOSED_CORRECTIONS` for the
   full, corrected accounting.
4. A symlink `baselines/explanation-quality-recsys/models/PGPR/datasets ->
   ../datasets` is created automatically by `PGPRModelAdapter.fit()` if missing — restores
   the directory layout `train_ml100k.sh` (the script AUDIT.md confirmed produced real
   results) already assumed; no source file touched.

## Known scope limitations

- **PGPR's train/test split is not rexbench's `DatasetBundle` split.** PGPR's own
  preprocessing pipeline has no parameterizable entry point for an arbitrary split — see
  `REGISTRY.md`. It's the one model not guaranteed to share exact train/test rows with every
  other model on the same dataset.
- **PGPR runs on ML100K/ML1M only.** The other 6 datasets need a KG-construction step that
  doesn't exist yet — see `REGISTRY.md` for the per-dataset breakdown and rough cost estimate.

## Running

```bash
cd measuring/                                     # repo root — configs use root-relative paths
rexbench run --config rexbench/configs/tier1.yaml  # EBPR family + PGPR only
rexbench run --config rexbench/configs/full.yaml   # + Tier 2 baselines + AR/KNN explainers
rexbench aggregate --run rexbench/outputs/<run_id>
rexbench stats --run rexbench/outputs/<run_id>
```

Each run writes `rexbench/outputs/<run_id>/`: `config.yaml` (copied verbatim),
`manifest.json` (git commit, library versions, hardware, disclosed corrections),
`results.parquet`/`.csv` (long format: dataset, model, explainer, metric, seed, k, value,
status), `failures.parquet`/`.csv` (one row per non-ok attempt, with exception type/message/
traceback or precondition measurements — never silently dropped).

## Verification status

Run without the merged environment installed:
```bash
cd rexbench && PYTHONPATH=src python3 -m pytest tests/test_config_schema.py tests/test_metrics_validate.py -v
```
14/14 passing — covers the config schema (including both real configs), and, most
importantly, the Gini fix (AUDIT.md section 5) against the exact degenerate inputs traced
there: all-zero scores and a single-user dict now return `NaN`, all-equal-nonzero scores
return `0.0`, and the normal case is unchanged.

`tests/test_dataset_bundle.py` and `tests/test_precondition_checks.py` require RecBole (a
transitive import of `rexfair` itself) or torch respectively — they were not executable in
this session (no merged environment was built here; environment provisioning was left as a
user action). Every adapter was written against the real, directly-read source of each
wrapped library (see each module's docstring for exact function/line references) but full
end-to-end training runs were not executed in this session — treat the adapters as
carefully-researched and internally consistent, not as execution-verified, until run once in
the built environment.
