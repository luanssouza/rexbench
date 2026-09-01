"""Seeding and the run manifest.

AUDIT.md 4.1 found every model on every dataset training under the same RecBole default
seed (2020) only because nothing ever overrode it — implicit, not explicit. seed_all()
makes the seed an explicit, config-driven value passed to every library that has its own
RNG, including RecBole's own Config/init_seed (AUDIT.md confirmed RecBole's own seeding is
reliable — two BPR/ml100k runs 6h apart were byte-identical — so this preserves that
mechanism rather than fighting it with a second, competing one).
"""
from __future__ import annotations

import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _git_info(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return None

    commit = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def _pip_freeze() -> list[str] | None:
    try:
        out = subprocess.run(["pip", "freeze"], capture_output=True, text=True, check=True)
        return sorted(out.stdout.strip().splitlines())
    except Exception:
        return None


# Applied and verified against the actually-executed source tree (baselines/
# explanation-quality-recsys/models/PGPR/*.py — confirmed via main.py's import graph, not
# the unused nested models/PGPR/models/PGPR/ copy). Every run manifest records these so the
# corrections are auditable, not silent.
DISCLOSED_CORRECTIONS = [
    {
        "target": "baselines/explanation-quality-recsys/models/PGPR/data_utils.py "
                   "(generate_review_dict, rating/timestamp parsing)",
        "change": "both `rating` and `timestamp` were gated on `dataset_name == 'ml1m'`, "
                   "hardcoding rating=0 and timestamp=<the rating value> for every other "
                   "dataset. Verified the ml100k review file has the identical 4-field "
                   "'uid pid rating timestamp' layout as ml1m (both fields genuinely "
                   "present at the same positions) — this was a real data-loading bug for "
                   "every non-ml1m dataset, not a format difference. Fix: parse both fields "
                   "unconditionally at their fixed positions.",
        "reference": "AUDIT.md section 3.1; corrected/expanded from the audit's original "
                     "finding after verifying the raw review file contents",
    },
    {
        "target": "baselines/explanation-quality-recsys/models/PGPR/test_agent.py (evaluate)",
        "change": "gender-fairness breakdown re-enabled against user2attribute/"
                   "attribute2name, which were already computed at the top of the function "
                   "but never consumed (was dead code that still printed nan/noOfUser=0 for "
                   "Male/Female while only Overall was ever populated)",
        "reference": "AUDIT.md section 3.2",
    },
    {
        "target": "baselines/explanation-quality-recsys/models/PGPR/test_agent.py:191 and "
                   "train_transe_model.py:67 (the only torch.load(...) call sites on the "
                   "actually-executed code path — verified; a broader initial estimate of "
                   "~7 sites included files in the unused nested tree and an EBPR method "
                   "never reached by this pipeline's fit() call sequence)",
        "change": "weights_only=False added explicitly (torch>=2.6 changed that default, "
                   "which would otherwise break loading these non-tensor state dicts; same "
                   "pattern already used in the predecessor pipeline's explanation module)",
        "reference": "rexbench plan, environment-unification feasibility check",
    },
]
# Note: the plan's initial feasibility check also flagged an np.float usage (removed in
# numpy>=1.24) at what was reported as transe_model.py:88. Verified before patching: that
# line only exists in the unused nested models/PGPR/models/PGPR/transe_model.py copy — the
# live transe_model.py already uses np.float64. No patch was needed or applied.


def build_manifest(config_dict: dict, seed: int, device: str) -> dict:
    versions: dict[str, str | None] = {}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "torch"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None

    torch_cuda = None
    try:
        import torch

        torch_cuda = bool(torch.cuda.is_available())
    except ImportError:
        pass

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "device_requested": config_dict.get("determinism", {}).get("device"),
        "device_resolved": device,
        "git": _git_info(REPO_ROOT),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "library_versions": versions,
        "torch_cuda_available": torch_cuda,
        "pip_freeze": _pip_freeze(),
        "disclosed_corrections": DISCLOSED_CORRECTIONS,
        "config": config_dict,
    }
