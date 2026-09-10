# Running rexbench on Lightning AI

No local GPU needed — a Lightning AI Studio works well here (RecBole/EBPR/PGPR were already
run this way earlier in this project, per the audit's `lightiningai/gpu-studio/` evidence).

Two things worth knowing before you start:
- EBPR's dense co-occurrence matrix (`create_explainability_matrix`) is built in **CPU RAM**,
  not on the GPU — pick a Studio machine with enough memory, not just a big GPU.
- PGPR's knowledge-graph data still has no verified download source (see `REGISTRY.md`) —
  bring your own copy if you have one.

## Steps

1. **Create a Studio** at lightning.ai. Start it on a CPU machine for setup (cheaper); you
   only need the GPU tier once you're actually training.

2. **Give the Studio access to your forks**, if `external/*` point at private repos — set up
   a GitHub personal access token or SSH key in the Studio's terminal before cloning.

3. **Clone with submodules**:
   ```bash
   git clone --recurse-submodules https://github.com/<you>/rexbench.git
   cd rexbench
   # if you forgot --recurse-submodules:
   git submodule update --init --recursive
   ```

4. **Build the merged environment**:
   ```bash
   conda create -n rexbench python=3.10 -y
   conda activate rexbench
   pip install -e .
   pip install -e external/recoxplainer/
   ```

5. **Switch the Studio to a GPU machine** (Studio settings/machine-type picker), then verify
   it's visible:
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   If this prints `False`, the machine switch didn't take effect yet — restart the Studio
   session and reactivate the conda env.

6. **Download the datasets** per [`data/README.md`](data/README.md) directly in the Studio
   terminal (e.g. `wget`/`curl` from each dataset's official source) into `data/raw/...` —
   downloading server-to-server inside the Studio is much faster than uploading from your
   laptop, especially for LastFM1K (2.4GB). `data/raw/` is gitignored by design, so this is a
   one-time step per Studio, not something that travels with the repo.

6b. **(Recommended) Bring your own pre-split data instead of re-splitting on the Studio.**
   Every config's datasets set `split.store_dir: data/splits/...` (see README.md's "Persisted
   splits" section) — if you already ran `rexbench split --config configs/tier1.yaml`
   locally, `rsync` the result up instead of trusting the Studio to re-derive byte-identical
   splits from its own copy of the raw data:
   ```bash
   rsync -avz data/splits/ my-studio:rexbench/data/splits/
   ```
   If you skip this, the first `rexbench run`/`rexbench split` on the Studio just computes
   and persists the split there itself — still fully reproducible (deterministic given the
   same raw file + config), just not guaranteed byte-identical to a split computed on a
   different machine/pandas version. Either way, once `data/splits/<name>/` exists anywhere,
   every later run against that `store_dir` reuses it verbatim rather than recomputing.

7. **Run the smoke test first**, before spending real compute on a full config:
   ```bash
   rexbench run --config configs/smoke_test.yaml
   ```
   Every model we have (EBPR family, PGPR, all 5 Tier-2 baselines, AR/KNN explainers) on
   every dataset, sampled down to ~100 users each (see `configs/smoke_test.yaml`'s comments)
   — this confirms the whole environment/pipeline actually runs end-to-end on the new
   Studio, in minutes rather than hours, before you commit to a real run. Note PGPR still
   trains on its full on-disk ml100k/ml1m data even here (it doesn't use the sample — a
   disclosed, pre-existing scope limitation, see `REGISTRY.md`), just with 1 epoch instead
   of 30/50, so it's faster but not tiny like everything else.

   Once that's clean, run for real:
   ```bash
   rexbench run --config configs/tier1.yaml        # EBPR family + PGPR only
   rexbench run --config configs/hpo_example.yaml  # + HPO search on EBPR
   ```

8. **Get your results out.** `outputs/<run_id>/{results,failures,trials}.parquet` and
   `manifest.json` live in the Studio's persistent storage (it survives stopping the Studio,
   unlike a plain ephemeral notebook) — use `rexbench aggregate`/`rexbench stats` to
   post-process in place, and pull files out via the Studio's file browser or `scp`/`rsync`
   if you have SSH access configured.

9. **Stop the Studio (or drop back to a CPU machine) when you're not actively training** —
   GPU time is billed, persistent storage is not, so there's no cost benefit to leaving a GPU
   machine running idle between runs.
