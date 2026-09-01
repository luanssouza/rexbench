# Registry of implementations NOT wired into rexbench

Every entry below is a real, audited implementation that `rexbench` deliberately does not
run. Kept documented, not deleted — the reasons these don't run are themselves a finding
(see "AUDIT.md" — the prior reproducibility audit this pipeline was built from; that
document lives outside this repository, referenced here by section for provenance).

| Implementation | Reason not wired in | Audit reference |
|---|---|---|
| `baselines/PGPR` | Pristine, unexecuted fork of upstream PGPR. Never adapted to any project dataset (clean `git status`, only upstream commits). Superseded by the vendored PGPR copy inside `baselines/explanation-quality-recsys/models/PGPR`, which rexbench's `PGPRModelAdapter` wraps instead. | AUDIT.md section 3 (baselines audit) |
| `baselines/ProtoMF` | Attempted repeatedly (4 successive tuning runs), crashed every time: a Ray Tune API version mismatch (`AttributeError: module 'ray.tune' has no attribute 'checkpoint_dir'`) plus a broken/inconsistent data path (`FileNotFoundError` on `user_ids.csv`, with a doubled path segment in some trials). No successful trial, no checkpoint, no results anywhere in the repo. Not in Tier 1 or Tier 2 scope. | AUDIT.md section 3.6 |
| `baselines/lod-personalized-recommender` | Ran cleanly, but only against the original authors' own demo dataset (hetrec2011-lastfm-2k) — every commit is authored by the upstream author; the researcher made zero commits and never adapted it to any project dataset. Reads as a literature-survey clone, not an active baseline. | AUDIT.md section 3 (baselines audit) |
| `baselines/rep-path-reasoning-recsys` (CAFE, UCPR) | CAFE and UCPR only reached the TransE knowledge-graph-embedding pretraining stage (`pretrained/ml1m/{cafe,ucpr}/transe/`) — no ml100k pretrained directory exists for either, and neither was ever trained to a full policy/symbolic model or evaluated. The same directory's own PGPR copy (a third, independent vendored copy of PGPR) trained successfully but was never merged with the ETD/SEP/LIR-augmented PGPR pipeline that actually produced published results — superseded, not integrated. | AUDIT.md section 3 (baselines audit) |
| `baselines/recoxplainer` | Byte-for-byte-diverged, stale duplicate of the real dependency. `git log` shows only clean upstream commits; the actively-patched fork with the memory/batching/NaN fixes rexbench actually needs lives at the top-level `recoxplainer/` (remote `luan`, uncommitted in-progress edits). Using this copy instead would silently reintroduce the NaN/batching bugs the top-level fork fixed. | AUDIT.md section 0 / 3 |
| RecBole `ItemKNN` | Fully implemented and importable (`AVAILABLE_MODELS` in `models/recbole_adapter.py`, ported from the predecessor pipeline's training code), trained repeatedly in the predecessor project (7 checkpoints across 5 dates), but never evaluated in either the predecessor pipeline or rexbench — commented out of every explanation config with no explanatory comment anywhere in the repo. Not part of your Tier 2 list either (Popularity/BPR/MultiVAE/NeuMF/SLIM only). Left out of rexbench for the same reason it was already excluded, undocumented as that reason is. | AUDIT.md sections 1.1, 3.9 |
| PGPR's ETD/SEP/LIR reranking variants (`ETDopt`, `LIRopt`, `SEPopt`, and their 4 combinations) | Beyond what Tier 1 asked for (base PGPR only). The chain that produced their published numbers is confirmed broken: `mesuaring-explainable-fairness/analysis/post_process_pgpr.ipynb` reads from `baselines/explanation-quality-recsys/paths_v5/` and `results_v5/`, and those directories no longer exist anywhere in the tree. Wiring these in would mean either accepting an unreproducible cached result or rebuilding the missing postprocessing step from scratch — scoped as future work, not attempted here. | AUDIT.md section 3.10 |
| LIME / SHAP / item-similarity explainers (predecessor project) | Explored in `mesuaring-explainable-fairness/explanation/{lime,shap,item_similarity_based,post_hoc_similarity}.ipynb`, never carried into the predecessor pipeline's explanation layer, and not requested for this pipeline (only AR/KNN were asked for). | AUDIT.md section 3 (baselines audit) |
| `RecBole-FairRec` / `Recbole-Debias` (predecessor project submodules) | Scaffolded (`.gitmodules` added) but never executed — no logs, no saved checkpoints in either submodule. Represent an alternative research direction (fix unfairness at training time) the project pivoted away from in favor of the current post-hoc-measurement approach `rexbench` implements. | AUDIT.md section 3 (baselines audit) |

## Datasets PGPR cannot currently run on

Not a missing-implementation issue — PGPR's own KG-relation registry (`baselines/explanation-quality-recsys/models/PGPR/utils.py`) only recognizes `ml100k`, `ml1m`, and `lastfm` as dataset names, and only `ml100k`/`ml1m` have the actual pre-linked KG data files populated. See the plan's per-dataset table and each dataset config's `kg.status` field for the full breakdown (`needs_construction` vs `unsupported`), reproduced here:

| Dataset | Status |
|---|---|
| Coat, Electronics, Amazon(ModCloth-family), RentTheRunway, AMBAR, LastFM1K | `needs_construction` — real categorical/entity side-info exists in the raw data (or, for LastFM1K, a relation *schema* already exists in code with no data files), but no dataset-specific relation-extraction code exists to turn it into a KG PGPR can consume. Estimated ~0.5–1 day of new engineering each. |
| Amazon Digital Music | `unsupported` — review text/ratings only, no structured entity or category fields to build a KG from without additional NLP work. |

## Known scope limitation: PGPR's train/test split

`PGPRModelAdapter` (`rexbench/src/rexbench/models/pgpr_adapter.py`) runs PGPR's own
pre-existing on-disk `datasets/{ml100k,ml1m}/train.txt`/`test.txt` split, not rexbench's
`DatasetBundle` split. PGPR's preprocessing pipeline (`preprocess.py`, `myutils.py`) is
hardcoded around dataset-name string constants and fixed relative paths with no
parameterizable entry point for an arbitrary split. Reconciling this — regenerating PGPR's
`train.txt`/`test.txt` from `DatasetBundle`, keyed through PGPR's own review-uid <-> KG-uid
mapping (`user_mappings.txt`) — is real, scoped follow-up work, not done in this pass.
This means PGPR is the one model in this pipeline not guaranteed to share the exact same
train/test rows as every other model on ml100k/ml1m — flag this explicitly in any
cross-model comparison table.

## Known gap: PGPR's KG relation data has no verified canonical download source

`data/raw/pgpr_kg/{ml100k,ml1m}/` (entity/relation files, `kg_final.txt`, `e_map.txt`,
`r_map.txt`, `train.txt`/`test.txt`) is a research-paper-specific derived artifact, not a
standard dataset release — it isn't tracked in the `explanation-quality-recsys` fork's git
history either (confirmed untracked there). `rexbench data fetch` documents manual placement
for it rather than a download URL, since no stable, verified external host was found for this
exact derived data. If you have (or know) a canonical source for it, add it to
`src/rexbench/data_fetch.py`'s registry.
