import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union


PathLike = Union[Path, str]


@lru_cache(maxsize=1)
def get_current_commit_hash(start_path: Optional[PathLike] = None) -> Optional[str]:
    """
    Return the current git commit hash for the repository containing start_path.

    Falls back to the KARMA_GIT_COMMIT environment variable when present and
    gracefully handles environments without git or repository metadata.
    """

    env_override = os.getenv("KARMA_GIT_COMMIT")
    if env_override:
        return env_override.strip() or None

    repo_root = _find_git_root(start_path)
    if repo_root is None:
        return None

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    return commit or None


def _find_git_root(start_path: Optional[PathLike]) -> Optional[Path]:
    """Locate the git repository root containing start_path, if any."""

    if start_path is None:
        path = Path.cwd()
    else:
        path = Path(start_path)
        if path.is_file():
            path = path.parent
    path = path.resolve()

    for candidate in [path, *path.parents]:
        git_dir = candidate / ".git"
        if git_dir.exists():
            return candidate
    return None
