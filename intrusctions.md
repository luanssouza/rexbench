1. Studio + environment (skip if already done):


git clone --recurse-submodules https://github.com/luanssouza/rexbench.git
cd rexbench
conda create -n rexbench python=3.10 -y && conda activate rexbench
pip install -e .
pip install -e external/recoxplainer/
Switch to a GPU machine-type once past setup; verify with python -c "import torch; print(torch.cuda.is_available())".

2. Sync the splits you just made — this is the new part:


rsync -avz data/splits/ my-studio:rexbench/data/splits/

cp -r /teamspace/uploads/data ./

That's data/splits/smoke_test/{ml100k,ml1m,coat,electronics,rentrunway,ambar,lastfm1k,amazon_digital_music}/, each with train.csv/val.csv/test.csv/split_meta.json — small files, no need to touch the Studio's scp/upload path for the actual raw datasets.

3. Raw data: only PGPR needs anything, and it's not the same data. configs/smoke_test.yaml points every dataset's raw_path at data/raw/..., but since the persisted split at data/splits/smoke_test/<name> already matches (same split/sample settings as smoke_test_local.yaml — raw_path itself isn't checked, that's the whole point), none of EBPR/UBPR/UEBPR/Pop/BPR/MultiVAE/NeuMF/SLIM/AR/KNN will ever touch data/raw/ on the Studio. The one exception is PGPR, which uses its own separate dataset format entirely (data/raw/pgpr_kg/{ml100k,ml1m}, symlinked in by pgpr_adapter.py) — if you don't have that on the Studio, PGPR just shows up as a precondition-unmet failure row, it won't block the rest of the smoke test.

4. Run it:


rexbench run --config configs/smoke_test.yaml

Use the standard configs/smoke_test.yaml on the Studio, not smoke_test_local.yaml — the local one's raw_paths point at ../datasets/..., which won't exist there and also isn't needed there.

5. Pull results back: outputs/<run_id>/{results,failures,trials}.parquet + manifest.json, via rexbench aggregate/rexbench stats in place or scp/rsync down. Stop the Studio (or drop to CPU) once done.

If you haven't created the Studio yet, want me to walk through step 1 in more detail, or is that part already set up?




If you already have a Studio set up from before:


cd rexbench
git pull                                    # picks up split persistence, stage-pgpr-kg, PGPR split-sharing fix
git submodule update --init --recursive     # submodule pointers unchanged this round, but harmless to run
No new Python dependencies were added (everything new uses stdlib — gzip, shutil), so no pip install changes needed.

If you haven't created a Studio yet, follow LIGHTNING_AI.md steps 1–5 (create Studio, clone with --recurse-submodules, build the conda env, switch to GPU, verify torch.cuda.is_available()).

Then, sync the data you already prepared locally (none of this travels via git — both are gitignored):


rsync -avz data/splits/ my-studio:rexbench/data/splits/
rsync -avz data/raw/pgpr_kg/ my-studio:rexbench/data/raw/pgpr_kg/

That's the data/splits/smoke_test/<name> splits for all 8 datasets, plus the ml100k/ml1m PGPR KG data you staged with stage-pgpr-kg earlier.

Run it:


rexbench run --config configs/smoke_test.yaml

This will use your persisted splits (no raw dataset downloads needed except for PGPR's KG data, which you've now synced), and PGPR will regenerate its own train.txt/test.txt from that same split automatically inside pgpr_runtime/ — nothing extra to do for that on the Studio side.

Get results back:


rexbench aggregate --run outputs/<run_id>   # or scp/rsync outputs/<run_id>/ down directly
Want me to check anything else before you kick this off — e.g. verify the config once more, or double check disk/memory sizing for the GPU machine type?


Ah — Lightning Studios only allow one conda env (the pre-existing default), so conda create is out. Just install directly into that default env instead:


conda env list          # confirms the default env's name and shows which one is active (marked with *)
Then, with that default env active (it should already be, or conda activate <name-from-above>):


cd rexbench
pip install -e .
pip install -e external/recoxplainer/
pip install kmeans_pytorch
pip install "ray[tune]"
pip install python-box

No env name needed beyond whatever conda env list showed you — everything else (the pinned numpy<2.0/torch>=2.0/recbole/etc. set from pyproject.toml) installs the same way regardless of which env it lands in.

For the batch Job command from our earlier discussion, swap rexbench for that real env name:

cd rexbench
conda run -n cloudspace --no-capture-output rexbench run --config configs/smoke_test.yaml

Can you paste what conda env list shows? I'll adjust the exact command once I know the name rather than guess at it.


Lightning AI's batch mechanism is their Jobs feature — runs a command non-interactively on a chosen machine, keeps going after you disconnect, and reuses your Studio's filesystem (so it sees your cloned repo and synced data as-is). Two ways to launch one, both from inside your Studio:

Via the CLI/SDK (what I have moderate confidence on, but double-check against lightning run job --help since exact flags shift between SDK versions):


lightning run job \
  --name rexbench-smoke-test \
  --machine A10G \
  --command "conda run -n rexbench --no-capture-output rexbench run --config configs/smoke_test.yaml"

smoke-test-26-09-11-v4