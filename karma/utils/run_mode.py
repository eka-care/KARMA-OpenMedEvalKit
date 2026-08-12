"""Dev/prod run-mode derivation (benchmarking spec B2). Never user-set, never hashed."""

import logging
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional

from karma.utils.git import _find_git_root

logger = logging.getLogger(__name__)

_MAIN_BRANCHES = ("main", "master")


def derive_run_mode(*, max_samples: Optional[int] = None) -> str:
    """``dev`` if the run is partial (max_samples) or code is off master head, else ``prod``."""

    if max_samples is not None:
        return "dev"
    if not _code_matches_main_head():
        return "dev"
    return "prod"


@lru_cache(maxsize=1)
def _code_matches_main_head() -> bool:
    repo_root = _find_git_root(Path(__file__))
    if repo_root is None:
        # No repo metadata (pip install / container image) — trust the deployment.
        return True

    head = _git(repo_root, "rev-parse", "HEAD")
    if head is None:
        return True

    if _git(repo_root, "status", "--porcelain", "--untracked-files=no"):
        return False

    for branch in _MAIN_BRANCHES:
        remote_head = _git(repo_root, "rev-parse", f"origin/{branch}")
        if remote_head:
            return head == remote_head

    return True


def _git(repo_root, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return result.stdout.strip() or None
