"""sys.path shims so adapters can import the vendored baseline implementations in place.

Nothing under external/ is copied or forked here — these are git submodules (see
.gitmodules), so this only makes those packages importable from the shared rexbench
environment described in README.md#environment. Two of the three submodules carry small,
disclosed compatibility/correctness patches on top of their upstream forks (not method
changes) — see README.md#disclosed-corrections and the `disclosed_corrections` list every
run manifest records.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../rexbench/

EXTERNAL = REPO_ROOT / "external"
RECOXPLAINER = EXTERNAL / "recoxplainer"  # submodule -> luanssouza/recoxplainer (fork)
EBPR_ROOT = EXTERNAL / "ebpr"  # submodule -> luanssouza/EBPR (fork)
EQR_ROOT = EXTERNAL / "explanation-quality-recsys"  # submodule -> luanssouza/explanation-quality-recsys (fork)
PGPR_ROOT = EQR_ROOT / "models" / "PGPR"  # the outer, actually-executed tree (see AUDIT.md / README)


def install() -> None:
    """Idempotently add every vendored package root to sys.path.

    EBPR_ROOT and PGPR_ROOT are added because their own modules use bare, repo-relative
    imports (e.g. EBPR's `from Code.EBPR_model import BPREngine`, PGPR's `from kg_env import
    BatchKGEnvironment`) rather than package-qualified ones — this is how those scripts were
    designed to be run, not a rexbench choice.
    """
    for p in (RECOXPLAINER, EBPR_ROOT, PGPR_ROOT, EQR_ROOT):
        sp = str(p)
        if p.exists() and sp not in sys.path:
            sys.path.insert(0, sp)


install()