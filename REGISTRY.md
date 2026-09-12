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

## RESOLVED: PGPR's train/test split now matches DatasetBundle

Previously, `PGPRModelAdapter` ran PGPR's own pre-existing on-disk `train.txt`/`test.txt`
split, not rexbench's `DatasetBundle` split — flagged as scoped follow-up work. Now fixed:
`fit()` regenerates PGPR's `train.txt`/`test.txt` (+ `.gz`) from
`dataset.train`+`dataset.val` (merged — PGPR has no validation concept and isn't wired into
HPO) and `dataset.test`, translated to PGPR's raw MovieLens ids via
`DatasetBundle.user_id_map`/`item_id_map` (verified these are the same ids PGPR's own
`mappings/user_mappings.txt`/`product_mappings.txt` key on). PGPR now shares the exact same
test rows as every other model on ml100k/ml1m, and — as a side effect — also respects
`SampleConfig` (smoke-test sampling), which it previously ignored.

Two things worth knowing:
- Items with no KG entity node (ml100k: 1424/1682 movies covered; ml1m: 3265/3706) are
  silently skipped by PGPR's own pre-existing `generate_labels`/`load_reviews` logic
  (`if product_idx not in id2kgid: continue`) — this gap already existed on PGPR's original
  split; regenerating the split doesn't introduce it, just applies the same handling.
- A real, previously-undiscovered bug was found and fixed along the way: PGPR's
  `DATASET_DIR[name]` resolves via a relative path (`'../../datasets/<name>'`) to
  `external/explanation-quality-recsys/datasets/<name>` — a stale, incomplete,
  *git-tracked* directory inside the submodule — not `models/PGPR/datasets/<name>` as the
  adapter's symlink previously (and incorrectly) assumed; that symlink was never actually
  read by anything. `fit()` now monkey-patches `DATASET_DIR[name]` to a rexbench-owned
  `pgpr_runtime/<name>/` directory instead, built via `_materialize_pgpr_dataset_dir` — see
  `pgpr_adapter.py`'s module docstring.

## Known gap: PGPR's KG relation data has no verified canonical download source

`data/raw/pgpr_kg/{ml100k,ml1m}/` (entity/relation files, `kg_final.txt`, `e_map.txt`,
`r_map.txt`, `train.txt`/`test.txt`) is a research-paper-specific derived artifact, not a
standard dataset release — it isn't tracked in the `explanation-quality-recsys` fork's git
history either (confirmed untracked there). `data/README.md` documents manual placement
for it rather than a download URL, since no stable, verified external host was found for this
exact derived data. If you have (or know) a canonical source for it, update that file.

## Correctness finding: two real bugs surfaced while pinning the `ebpr` submodule

Found while updating the `external/ebpr` submodule pin from `5ac867e` to the `reproducibility`
branch's current tip (`a627034`) — not something the earlier audit caught, since it wasn't in
scope there. Both affect the EBPR/UBPR/UEBPR reproduction specifically:

1. **`Code/engine_EBPR.py`'s `train_an_epoch` had a debug `break` after the first batch of
   every epoch, present at the commit this pipeline was originally built against (`5ac867e`)**:
   ```python
   print("Isso aqui é um teste!")
   print(user)
   break
   loss = self.train_single_batch_EBPR(...)   # unreachable
   ```
   This means training at that commit processed exactly one batch per "epoch," not the full
   dataset — degenerate training, not a real reproduction, for however long that commit was in
   use. Fixed (commented out, not removed) at the current tip `a627034`, which is what the
   submodule is now pinned to. **If you have any results generated by hand-running EBPR at or
   before `5ac867e`, they should be treated as invalid and re-run.**

2. **`Code/metrics.py`'s `cal_map_at_k()` is a stub that always returns `10.999`** — the real
   computation (`mapk(actual, predicted, k=top_k)`) is commented out below it. Still present at
   the current tip (`a627034`); not something this pipeline's disclosed corrections touched
   (it's in `ebpr`, not `explanation-quality-recsys`). This method is only reached when
   `config['loo_eval'] == False` (the "explicit"/MAP evaluation path in
   `Engine.evaluate()`) — rexbench's `EBPRModelAdapter` now defaults `loo_eval=True`
   specifically to avoid this stub (see `models/ebpr_adapter.py`'s comment at the `loo_eval`
   config key), which also happens to be the only mode `fit()`'s `evaluate()`/`save_implicit()`
   call sequence was actually written correctly for — see finding 3.

3. **rexbench's own bug, now fixed**: `EBPRModelAdapter._build_config` defaulted
   `loo_eval=False`, but `fit()` unconditionally unpacks `engine.evaluate()`'s return as the
   `loo_eval=True` shape (`ndcg, hr, mep, ...`) and calls `save_implicit(...)`. With the old
   `False` default this would have silently bound the stubbed `10.999` MAP value to this
   adapter's `ndcg` variable and invoked the wrong save method entirely (`save_implicit` checks
   `best_performance[0]`, `save_explicit` checks `best_performance[1]` — not interchangeable).
   Never actually run this way (no execution happened in the session this pipeline was built
   in), but would have silently produced meaningless best-checkpoint selection for every
   EBPR/UBPR/UEBPR run. Fixed by changing the default to `loo_eval=True`, which both matches
   `fit()`'s actual call sequence and avoids the `cal_map_at_k` stub.
