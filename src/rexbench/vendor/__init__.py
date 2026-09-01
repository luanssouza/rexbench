"""sys.path shims so adapters can import the audited implementations in place.

Nothing under baselines/, rexfair/, RecBole/, or recoxplainer/ is copied or forked here —
this only makes those packages importable from the shared rexbench environment described
in README.md#environment. Two vendored source files carry small, disclosed compatibility
patches (not method changes) — see README.md#disclosed-corrections and the
`disclosed_corrections` list every run manifest records.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

REXFAIR = REPO_ROOT / "rexfair"
RECOXPLAINER = REPO_ROOT / "recoxplainer"  # the actively-patched fork, not baselines/recoxplainer
EBPR_ROOT = REPO_ROOT / "baselines" / "EBPR"
EQR_ROOT = REPO_ROOT / "baselines" / "explanation-quality-recsys"
PGPR_ROOT = EQR_ROOT / "models" / "PGPR"  # the outer, actually-executed tree (see AUDIT.md / README)


def install() -> None:
    """Idempotently add every vendored package root to sys.path.

    EBPR_ROOT and PGPR_ROOT are added because their own modules use bare, repo-relative
    imports (e.g. EBPR's `from Code.EBPR_model import BPREngine`, PGPR's `from kg_env import
    BatchKGEnvironment`) rather than package-qualified ones — this is how those scripts were
    designed to be run, not a rexbench choice.
    """
    for p in (REXFAIR, RECOXPLAINER, EBPR_ROOT, PGPR_ROOT, EQR_ROOT):
        sp = str(p)
        if p.exists() and sp not in sys.path:
            sys.path.insert(0, sp)


install()
