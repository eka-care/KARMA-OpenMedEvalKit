"""Metric-level language resolution: lenient coercion + Indic normalizer fallback.

Hash identity is strict (karma.utils.language.normalize_language); metrics are
lenient — an unsupported language degrades to generic 'en' handling, never a crash.
"""

import logging
from typing import Optional, Tuple

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

from karma.utils.language import metric_language

logger = logging.getLogger(__name__)


def resolve_indic_normalizer(
    value: Optional[str], **normalizer_kwargs
) -> Tuple[str, Optional[object]]:
    """Return (language, normalizer) for a metric. Unknown or normalizer-less
    languages resolve to ('en', None) — the generic path."""

    language = metric_language(value)
    if language == "en":
        return language, None
    try:
        return language, IndicNormalizerFactory().get_normalizer(
            language, **normalizer_kwargs
        )
    except Exception:
        logger.warning(
            "No Indic normalizer for %r; using generic 'en' handling.", language
        )
        return "en", None
