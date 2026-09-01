# Log of changes to pre-existing files

Scope of this file: only files that **existed before this session** and were modified by me.
Every other file touched in the last turn (everything under `rexbench/`) was newly created —
there is no "original version" of those to log or preserve, so they're not covered here.

This directory (`rexbench/old/`) mirrors the repo-root-relative path of each modified file
and holds its exact pre-edit content, pulled from `git show HEAD:<path>` in the
`baselines/explanation-quality-recsys` repo (confirmed identical to the pre-edit blob before
writing — see verification note at the bottom). Diff any current file against its `old/`
counterpart to see exactly what changed:

```bash
diff rexbench/old/baselines/explanation-quality-recsys/models/PGPR/data_utils.py \
     baselines/explanation-quality-recsys/models/PGPR/data_utils.py
```

All three files below were approved by you before editing (see the conversation's
"PGPR bug handling" and "Env unification" questions) and are already summarized in
`rexbench/README.md` and recorded in every run's `manifest.json` via
`rexbench/src/rexbench/core/determinism.py`'s `DISCLOSED_CORRECTIONS` list. This file gives
the full line-level diff for each, which those two only summarize.

---

## 1. `baselines/explanation-quality-recsys/models/PGPR/data_utils.py`

**What changed** — `AmazonDataset`'s review-parsing loop (inside the method building
`self.review`), 2 lines:

```diff
-            rating = int(arr[2]) if self.dataset_name == "ml1m" else 0
-            timestamp = int(arr[3]) if self.dataset_name == "ml1m" else int(arr[2])
+            # DISCLOSED CORRECTION (rexbench, see AUDIT.md section 3.1 and
+            # rexbench/REGISTRY.md): both lines below were gated on
+            # `dataset_name == "ml1m"`, hardcoding rating=0 and timestamp=<the rating value>
+            # for every other dataset. Verified the ml100k review file has the identical
+            # 4-field `uid pid rating timestamp` format as ml1m (both fields are genuinely
+            # present at the same positions), so this was a real data-loading bug, not a
+            # dataset-format difference — not gating on dataset_name at all is the correct
+            # fix for every dataset using this same field layout.
+            rating = int(arr[2])
+            timestamp = int(arr[3])
```

**Why**: `AUDIT.md` section 3.1 originally flagged only the `rating` line — for any dataset
other than the literal string `"ml1m"`, every review was hardcoded to `rating=0`
("negative"), confirmed empirically (ml100k's own log showed 0 positive / 73,625 negative
reviews). Before applying the approved fix, I checked the actual gzipped review file content
(`datasets/ml100k/train.txt.gz`) to confirm what ml100k's real format is, since "fix it to use
the real rating" is only possible if a real rating value exists at that file position. It
does — ml100k's review lines are 4 fields (`uid pid rating timestamp`), identical in shape to
ml1m's. That check also surfaced a second, previously undisclosed bug in the very next line:
`timestamp` was being set to `arr[2]` (the rating value, e.g. `2` or `3`) instead of `arr[3]`
(the real Unix timestamp, e.g. `881514683`) for every non-ml1m dataset — the same
dataset-name-gating mistake applied to the adjacent field. I fixed both lines together since
they're the same root cause, and disclosed the expanded scope explicitly (this is broader
than what "fix the rating threshold" literally covered when you approved it — flagging that
here rather than treating it as covered by the earlier approval).

## 2. `baselines/explanation-quality-recsys/models/PGPR/test_agent.py`

**What changed** — two separate edits to the same file:

**(a) `evaluate()`: removed a stray leftover comment** (a duplicate, already-commented-out
re-fetch of `get_user2gender`, redundant with the same call already active at the top of the
function):
```diff
-    #uid2gender, gender2name = get_user2gender(dataset_name)
     test_user_idxs = list(test_user_products.keys())
```

**(b) `evaluate()`: re-enabled the gender-stratified metrics block**:
```diff
-        # Based on attribute
-#        attribute_val = uid2gender[uid]
-#        gender = gender2name[attribute_val]
+        # DISCLOSED CORRECTION (rexbench, see AUDIT.md section 3.2 and
+        # rexbench/REGISTRY.md): the gender-stratified block below was commented out while
+        # the printing loop at the end of this function still iterated Male/Female groups,
+        # so those groups always printed nan/noOfUser=0 rather than being computed at all.
+        # Re-enabled against user2attribute/attribute2name, which were already being
+        # computed at the top of this function (line 26) but never consumed. Users with no
+        # gender mapping are skipped for the stratified groups (still counted in Overall).
         all = "Overall"
-
-        # According to gender
- #       metrics.ndcg[gender].append(ndcg)
-#        metrics.recall[gender].append(recall)
- #       metrics.precision[gender].append(precision)
- #       metrics.hr[gender].append(hit)
+        attribute_val = user2attribute.get(uid)
+        if attribute_val is not None:
+            gender = attribute2name[attribute_val]
+            metrics.ndcg[gender].append(ndcg)
+            metrics.recall[gender].append(recall)
+            metrics.precision[gender].append(precision)
+            metrics.hr[gender].append(hit)
```

**Why**: `AUDIT.md` section 3.2 found this exact block commented out while the function's
final printing loop still iterated `Male`/`Female` groups unconditionally, producing
`Gender group Male, noOfUser=0, PGPR ndcg=nan` in the logs — looks like a completed run,
computes nothing. `user2attribute`/`attribute2name` were already being computed by an active
(uncommented) call at the top of the function and simply never consumed. Re-enabling this
does not change the RL/path-reasoning method itself, only what gets tallied per demographic
group from its output.

**(c) `predict_paths()`: `torch.load` compatibility fix**:
```diff
-    pretrain_sd = torch.load(policy_file, map_location=torch.device('cpu'))
+    # DISCLOSED CORRECTION (rexbench): weights_only=False added explicitly — torch>=2.6
+    # changed that default, which would otherwise break loading this non-tensor state dict.
+    pretrain_sd = torch.load(policy_file, map_location=torch.device('cpu'), weights_only=False)
```

**Why**: torch >=2.6 changed `torch.load`'s default `weights_only` to `True`, which rejects
checkpoints containing non-tensor Python objects (this policy checkpoint does). Same fix
already present, for the same reason, in the predecessor pipeline's explanation module.
No numeric/behavioral change — only which torch versions can load the file.

## 3. `baselines/explanation-quality-recsys/models/PGPR/train_transe_model.py`

**What changed** — `extract_embeddings()`, one line:
```diff
-    state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
+    # DISCLOSED CORRECTION (rexbench): weights_only=False added explicitly — torch>=2.6
+    # changed that default, which would otherwise break loading this non-tensor state dict.
+    state_dict = torch.load(model_file, map_location=lambda storage, loc: storage, weights_only=False)
```

**Why**: same `torch.load` compatibility issue and fix as (c) above, at the other call site
on the actually-executed code path.

## What I checked but did NOT change

An initial estimate (given to you before applying anything) said ~7 `torch.load` sites and
one `np.float` usage needed patching. Before touching any of them I re-verified each against
`main.py`'s actual import graph and found most were in
`models/PGPR/models/PGPR/` — an unused, never-imported nested copy of the same files — or in
an EBPR method (`baselines/EBPR/Code/utils.py:10`'s `resume_checkpoint`) that this pipeline's
adapter never calls. Only the two sites listed above are on the code path
`rexbench/src/rexbench/models/pgpr_adapter.py` actually executes, so only those two were
patched. Nothing in `models/PGPR/models/PGPR/` or `baselines/EBPR/` was touched.

## Not a file edit, but a related filesystem change

`baselines/explanation-quality-recsys/models/PGPR/datasets` — a **new symlink** (not a
modification of an existing file, so there's no "old" version to log) pointing to
`../datasets` (i.e. `baselines/explanation-quality-recsys/datasets`, where the real KG data
lives). `models/PGPR/datasets` didn't exist on disk before this session even though
`train_ml100k.sh` (the script that produced `AUDIT.md`'s confirmed successful PGPR runs)
assumes it does. This restores that assumed structure; it's currently untracked in
`baselines/explanation-quality-recsys`'s git status (`?? models/PGPR/datasets`) — remove it
with `rm baselines/explanation-quality-recsys/models/PGPR/datasets` if you'd rather not keep
it, though `rexbench`'s `PGPRModelAdapter.fit()` will recreate it automatically on the next
run if missing.

## What was NOT touched despite showing up in `git status`

`git status` in `baselines/explanation-quality-recsys`, `baselines/EBPR`, and top-level
`recoxplainer/` all show additional uncommitted changes (modified `.gitignore` files, staged
CSVs, notebook diffs, untracked data directories, `requirements_new.txt`, etc.). None of
those were made by me this session — they predate it (consistent with `AUDIT.md`'s own
description of these as actively-edited, uncommitted research repos). I'm calling this out
explicitly so you don't mistake pre-existing dirty state in those repos for something from
this conversation.

---

**Verification**: every file in this `old/` directory was confirmed byte-identical to
`git show HEAD:<path>` at the time it was written (`diff` against the git blob produced no
output for all three files) — these are exact pre-edit originals, not reconstructions from
memory.
