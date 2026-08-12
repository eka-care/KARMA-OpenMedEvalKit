"""Single global language normalization (datasets spec D2). Everything language-shaped
flows through normalize_language() — datasets, benchmark fallback, metrics."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ISO 639-1 (or closest short code) canonical set.
_CANONICAL = {
    "as", "bn", "brx", "doi", "en", "gu", "hi", "kn", "kok", "ks",
    "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd",
    "ta", "te", "ur",
    # Non-Indic languages used by retrieval/protocol datasets.
    "af", "de", "es", "fr", "pt", "xh", "zu",
}

_ALIASES = {
    "assamese": "as",
    "bengali": "bn",
    "bodo": "brx",
    "dogri": "doi",
    "english": "en",
    "en-in": "en",
    "en-us": "en",
    "eng": "en",
    "gujarati": "gu",
    "hindi": "hi",
    "kannada": "kn",
    "konkani": "kok",
    "kashmiri": "ks",
    "maithili": "mai",
    "malayalam": "ml",
    "manipuri": "mni",
    "marathi": "mr",
    "nepali": "ne",
    "odia": "or",
    "oriya": "or",
    "punjabi": "pa",
    "sanskrit": "sa",
    "santali": "sat",
    "sindhi": "sd",
    "tamil": "ta",
    "telugu": "te",
    "urdu": "ur",
    "afrikaans": "af",
    "german": "de",
    "spanish": "es",
    "french": "fr",
    "portuguese": "pt",
    "xhosa": "xh",
    "zulu": "zu",
}


_ISO_TO_NAME = {
    "as": "assamese",
    "bn": "bengali",
    "brx": "bodo",
    "doi": "dogri",
    "en": "english",
    "gu": "gujarati",
    "hi": "hindi",
    "kn": "kannada",
    "kok": "konkani",
    "ks": "kashmiri",
    "mai": "maithili",
    "ml": "malayalam",
    "mni": "manipuri",
    "mr": "marathi",
    "ne": "nepali",
    "or": "odia",
    "pa": "punjabi",
    "sa": "sanskrit",
    "sat": "santali",
    "sd": "sindhi",
    "ta": "tamil",
    "te": "telugu",
    "ur": "urdu",
    "af": "afrikaans",
    "de": "german",
    "es": "spanish",
    "fr": "french",
    "pt": "portuguese",
    "xh": "xhosa",
    "zu": "zulu",
}


def language_name(iso_code: str) -> str:
    """Full lowercase name for an ISO code — for sources whose configs use names (e.g. indicvoices)."""

    code = normalize_language(iso_code)
    return _ISO_TO_NAME[code]


def normalize_language(value: Optional[str]) -> Optional[str]:
    """Normalize any language spelling to its ISO short code. Always call — idempotent.

    Raises ValueError on unknown values (strict by decision: a typo must not
    mint a new config_hash).
    """

    if value is None:
        return None
    key = value.strip().lower()
    if not key:
        return None
    if key in _CANONICAL:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(
        f"Unknown language '{value}'. Known codes: {sorted(_CANONICAL)}; "
        f"add new languages to karma/utils/language.py."
    )


def sample_language(value: Optional[str]) -> Optional[str]:
    """Per-row language from dataset metadata: ''/'unknown' → None, else strict ISO."""

    if not value or value == "unknown":
        return None
    return normalize_language(value)


def metric_language(value: Optional[str], default: str = "en") -> str:
    """Lenient coercion for metric-level hints: unknown values fall back to the
    generic default instead of failing a finished run (hash identity stays strict)."""

    try:
        return normalize_language(value) or default
    except ValueError:
        logger.warning(
            "Language %r has no metric-level support; using generic %r handling.",
            value,
            default,
        )
        return default
