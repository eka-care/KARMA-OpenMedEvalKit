"""Retry-with-backoff for transient remote-API failures (429/5xx/network)."""

import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS = {429, 500, 502, 503, 504, 529}
_TRANSIENT_MARKERS = (
    "429",
    "rate limit",
    "rate_limit",
    "too many requests",
    "500",
    "502",
    "503",
    "504",
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "temporarily unavailable",
)


def is_transient_error(exc: BaseException) -> bool:
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and value in _TRANSIENT_STATUS:
            return True
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


def call_with_retry(
    fn: Callable[[], Any],
    *,
    max_attempts: int = 4,
    base_delay: float = 10.0,
    max_delay: float = 30.0,
    is_retryable: Optional[Callable[[BaseException], bool]] = None,
    description: str = "remote call",
) -> Any:
    """Run ``fn``; on a transient error wait min(base_delay * 2**attempt,
    max_delay) — 10s, 20s, 30s by default — and retry.

    Non-retryable errors raise immediately; the last transient error raises
    after ``max_attempts``.
    """
    check = is_retryable or is_transient_error
    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            if not check(exc):
                raise
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = min(base_delay * (2**attempt), max_delay)
                logger.warning(
                    "%s failed transiently (attempt %d/%d), retrying in %.1fs: %s",
                    description, attempt + 1, max_attempts, delay, str(exc)[:200],
                )
                time.sleep(delay)
    raise last_exc
