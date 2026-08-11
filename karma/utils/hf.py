"""HF hub revision resolution for model version pinning (models spec M5)."""

import logging
import os
import re
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

_HUB_ID_RE = re.compile(r"^[\w.\-]+/[\w.\-]+$")


def looks_like_hub_id(model_id: Optional[str]) -> bool:
    if not model_id:
        return False
    if os.path.exists(model_id):
        return False
    return bool(_HUB_ID_RE.match(model_id))


@lru_cache(maxsize=32)
def resolve_hf_revision(model_id: str) -> Optional[str]:
    """Current commit sha for a hub model — provenance + load pin only, never hashed.

    Metadata call only; returns None on any failure or when HF_HUB_OFFLINE is set.
    """

    if not looks_like_hub_id(model_id):
        return None
    if os.getenv("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes"):
        logger.info("HF_HUB_OFFLINE set — skipping revision resolution for %s", model_id)
        return None
    try:
        from huggingface_hub import HfApi

        sha = HfApi().model_info(model_id).sha
    except Exception as exc:
        logger.warning("Could not resolve HF revision for %s: %s", model_id, exc)
        return None
    if sha:
        logger.info("Pinned HF revision for %s: %s", model_id, sha)
    return sha or None
