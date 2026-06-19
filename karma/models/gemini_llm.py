"""Gemini text-generation provider for KARMA.

Extends :class:`BaseModel` and wraps the Google GenAI SDK
(``google-genai``) for text-only chat completion / judge calls.

Why a dedicated LLM provider (vs. reusing ``GeminiASR.generate_text``):

* Retry wrapper with exponential backoff (30s -> 60s -> 120s, max 3
  attempts) for transient errors (429 rate-limit, 500 / 503 / 504
  server-side, deadline-exceeded). Permanent errors (401 / 403 / 404 /
  bad model name) are NOT retried.
* Registers ``gemini-2.5-pro`` and ``gemini-2.5-flash`` directly as
  ``TEXT_GENERATION`` variants. Note that
  :mod:`karma.models.gemini_asr` also registers these same KARMA names
  as ``AUDIO_RECOGNITION`` variants; because :func:`pkgutil.iter_modules`
  imports modules alphabetically, ``gemini_llm`` loads AFTER
  ``gemini_asr`` and its registrations overwrite the ASR entries. This
  is intentional: the ASR class already routed text inputs through
  ``generate_text``, so there was no wire-level dedicated Gemini LLM
  path. Users needing the audio behaviour can still instantiate
  :class:`GeminiASR` directly, and the ``gemini-2.5-flash-lite`` and
  other ASR-only IDs remain untouched.
* API key: prefers ``GEMINI_API_KEY`` (the current Google-recommended
  env var name), with fallbacks to ``GOOGLE_AI_API_KEY`` and
  ``GOOGLE_API_KEY`` for parity with the existing ASR provider.

Judge compatibility: the MedCaseReasoning judge metrics call
``self.model.run([eval_input])[0]`` and expect a string. ``run()`` here
returns ``List[str]`` exactly like :class:`OpenAILLM.run`, and
:meth:`postprocess` strips each element, so passing an instance of
this class as a judge works without any adapter code.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.data_models.model_meta import ModalityType, ModelMeta, ModelType
from karma.models.base_model_abs import BaseModel
from karma.registries.model_registry import register_model_meta

logger = logging.getLogger(__name__)


# HTTP status codes that indicate a transient condition worth retrying.
# 408 = request timeout, 429 = rate limit, 500/502/503/504 = server-side.
_TRANSIENT_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})

# HTTP status codes that indicate a permanent client-side failure; we do
# NOT retry these (wrong key, revoked key, bad model ID, bad request).
_PERMANENT_HTTP_CODES = frozenset({400, 401, 403, 404})


def _is_transient_genai_error(exc: BaseException) -> bool:
    """Classify a Google GenAI SDK exception as transient or permanent.

    Transient conditions are retried with exponential backoff; permanent
    ones are raised immediately so callers see the real failure instead
    of three-wait-budget-exhausted messages.
    """
    # SDK-structured errors carry a numeric HTTP code on ``.code``.
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        if code in _TRANSIENT_HTTP_CODES:
            return True
        if code in _PERMANENT_HTTP_CODES:
            return False
        # ServerError is 5xx by SDK convention; ClientError is 4xx.
        if isinstance(exc, genai_errors.ServerError):
            return True
        if isinstance(exc, genai_errors.ClientError):
            return False
        # Unknown code on APIError: fall through to string inspection.

    # Network-layer and deadline errors from the underlying httpx/grpc
    # stack surface as plain exceptions; match on the message text.
    message = str(exc).lower()
    transient_markers = (
        "deadline exceeded",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "temporarily unavailable",
        "service unavailable",
        "rate limit",
        "resource_exhausted",
        "unavailable",
    )
    return any(marker in message for marker in transient_markers)


class GeminiLLM(BaseModel):
    """Google Gemini text-generation provider.

    Accepts :class:`DataLoaderIterable` inputs (text / conversation +
    optional system prompt), runs them through ``generate_content`` in
    parallel via a thread pool, and returns a list of stripped strings.

    Tool-calling / function-calling is intentionally NOT implemented here
    — judge usage (the primary target) never calls tools, and the
    Google SDK's tool surface differs enough from OpenAI's to warrant a
    separate provider extension if/when an agentic Gemini path is needed.
    """

    # Retry budget mirrors ``SarvamLLM._RETRY_*`` constants.
    _RETRY_MAX_ATTEMPTS = 3
    _RETRY_INITIAL_DELAY = 30
    _RETRY_BACKOFF_FACTOR = 2

    def __init__(
        self,
        model_name_or_path: str = "gemini-2.5-pro",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_output_tokens: int = 8192,
        max_workers: int = 10,
        thinking_budget: Optional[int] = None,
        **kwargs,
    ):
        """Initialise the Gemini LLM client.

        Args:
            model_name_or_path: Google-published Gemini model ID, e.g.
                ``gemini-2.5-pro`` or ``gemini-2.5-flash``.
            api_key: API key; if ``None`` falls back to ``GEMINI_API_KEY``,
                ``GOOGLE_AI_API_KEY``, then ``GOOGLE_API_KEY``.
            temperature: Sampling temperature. 0.0 for deterministic
                judge behaviour.
            top_p: Nucleus sampling. 1.0 by default (no truncation).
            max_output_tokens: Response cap. Gemini 2.5 supports large
                outputs; 8192 is comfortable for judge and generation use.
            max_workers: Thread-pool concurrency for parallel ``run``
                dispatch.
            thinking_budget: Optional Gemini "thinking" token budget.
                ``None`` defers to the model default; ``0`` disables
                the thinking step entirely on 2.5+ models.
            **kwargs: Forwarded to :class:`BaseModel` (``device``,
                ``sleep_duration``, ...).
        """
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)

        self.model_id = model_name_or_path
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_AI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Gemini API key must be provided either as the api_key "
                "argument or via the GEMINI_API_KEY (preferred), "
                "GOOGLE_AI_API_KEY, or GOOGLE_API_KEY environment variable."
            )

        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.max_workers = max_workers
        self.thinking_budget = thinking_budget

        self.client: Optional[genai.Client] = None
        self.load_model()

    # ------------------------------------------------------------------
    # Client bootstrap
    # ------------------------------------------------------------------
    def load_model(self, **kwargs) -> None:
        """Instantiate the Google GenAI client.

        Idempotent: the registry calls :func:`load_model` again after
        ``__init__`` has already done so (see ``ModelRegistry.get_model``).
        Re-running is cheap — :class:`genai.Client` just holds HTTP
        session state — but we skip construction when already loaded.
        """
        if self.is_loaded and self.client is not None:
            return
        try:
            self.client = genai.Client(api_key=self.api_key)
            self.is_loaded = True
            logger.info("Gemini LLM client initialised with model=%s", self.model_id)
        except Exception as e:
            logger.error("Failed to initialise Gemini client: %s", e)
            raise RuntimeError(f"Failed to initialise Gemini client: {e}") from e

    # ------------------------------------------------------------------
    # Preprocess: DataLoaderIterable -> per-sample API kwargs
    # ------------------------------------------------------------------
    def preprocess(
        self, inputs: List[DataLoaderIterable], **kwargs
    ) -> List[Dict[str, Any]]:
        """Shape each :class:`DataLoaderIterable` into an API call payload.

        Returns one dict per sample with keys ``model``, ``contents``,
        and ``config`` suitable for ``client.models.generate_content``.
        Samples without any input text are silently skipped (matches
        :class:`OpenAILLM.preprocess` behaviour).
        """
        processed: List[Dict[str, Any]] = []
        for item in inputs:
            contents: List[types.Content] = []

            # Prefer explicit conversation turns; fall back to .input.
            if item.conversation and item.conversation.conversation_turns:
                for turn in item.conversation.conversation_turns:
                    # Gemini roles are "user" / "model"; OpenAI-style
                    # "assistant" collapses to "model", "system" turns
                    # embedded in the conversation are hoisted into
                    # system_instruction below.
                    role = turn.role
                    if role in ("assistant", "model"):
                        mapped_role = "model"
                    elif role == "system":
                        # Treat an in-conversation system turn as a
                        # user-side instruction prefix to avoid silently
                        # dropping it; most callers pass system prompts
                        # via ``item.system_prompt`` instead.
                        mapped_role = "user"
                    else:
                        mapped_role = "user"
                    contents.append(
                        types.Content(
                            role=mapped_role,
                            parts=[types.Part.from_text(text=turn.content)],
                        )
                    )
            elif item.input:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=item.input)],
                    )
                )
            else:
                logger.warning(
                    "No input or conversation data found for Gemini sample; "
                    "skipping."
                )
                continue

            config = self._build_generation_config(
                system_prompt=item.system_prompt
            )
            processed.append(
                {
                    "model": self.model_id,
                    "contents": contents,
                    "config": config,
                }
            )

        return processed

    def _build_generation_config(
        self, system_prompt: Optional[str] = None
    ) -> types.GenerateContentConfig:
        """Construct the per-call ``GenerateContentConfig``.

        Honours ``temperature``, ``top_p``, ``max_output_tokens`` and
        — when set on the instance — ``thinking_budget``. System prompt,
        if provided on the sample, becomes ``system_instruction``.
        """
        config_kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        return types.GenerateContentConfig(**config_kwargs)

    # ------------------------------------------------------------------
    # Retry wrapper + single-call path
    # ------------------------------------------------------------------
    def _call_api_with_retry(self, **call_kwargs: Any):
        """Call ``generate_content`` with transient-only retry.

        Retry triggers:
            * SDK ``APIError`` whose HTTP status is in
              :data:`_TRANSIENT_HTTP_CODES`, or whose class is
              :class:`ServerError`.
            * Network/deadline errors whose message matches the transient
              markers in :func:`_is_transient_genai_error`.

        Non-retryable conditions (bad API key, bad model ID, malformed
        request, 4xx) raise immediately.

        Returns the raw SDK response object; callers extract ``.text``.
        """
        delay = self._RETRY_INITIAL_DELAY
        last_exc: Optional[BaseException] = None

        for attempt in range(1, self._RETRY_MAX_ATTEMPTS + 1):
            try:
                return self.client.models.generate_content(**call_kwargs)
            except Exception as exc:
                last_exc = exc
                if not _is_transient_genai_error(exc):
                    logger.error(
                        "Gemini permanent error (model=%s): %s",
                        self.model_id,
                        exc,
                    )
                    raise
                if attempt >= self._RETRY_MAX_ATTEMPTS:
                    logger.error(
                        "Gemini retry budget exhausted after %d attempts "
                        "(model=%s, last error=%s).",
                        attempt,
                        self.model_id,
                        exc,
                    )
                    raise
                logger.warning(
                    "Gemini transient error %d/%d (model=%s, error=%s); "
                    "sleeping %ds before retry.",
                    attempt,
                    self._RETRY_MAX_ATTEMPTS - 1,
                    self.model_id,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay *= self._RETRY_BACKOFF_FACTOR

        # Defensive — the loop either returns or raises.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Gemini retry loop exited without a response")

    def _make_single_call(self, api_input: Dict[str, Any]) -> str:
        """Execute one ``generate_content`` call and return text.

        Any exception (transient-exhausted or permanent) is caught and
        surfaced as an ``Error: ...`` string so a single failing sample
        does not kill the whole batch — mirrors
        :meth:`OpenAILLM._make_single_call`.
        """
        try:
            response = self._call_api_with_retry(**api_input)
        except Exception as e:
            logger.error("Failed to generate text with Gemini: %s", e)
            return f"Error: {e}"

        # The SDK exposes ``.text`` when the response has a single
        # text part. Fall back to joining all text parts if ``.text``
        # returns None (e.g. safety-filtered or multi-part responses).
        text = getattr(response, "text", None)
        if text:
            return text
        try:
            candidates = getattr(response, "candidates", None) or []
            parts_text: List[str] = []
            for cand in candidates:
                content = getattr(cand, "content", None)
                for part in getattr(content, "parts", None) or []:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        parts_text.append(part_text)
            return "\n".join(parts_text)
        except Exception as e:  # pragma: no cover - defensive only
            logger.warning("Unexpected Gemini response shape: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Run: parallel dispatch + postprocess
    # ------------------------------------------------------------------
    def run(self, inputs: List[DataLoaderIterable], **kwargs) -> List[str]:
        """Run Gemini text generation on the batch in parallel.

        Preserves input order; returns ``List[str]`` of the same length
        as ``processed_inputs`` (samples skipped by :meth:`preprocess`
        are dropped, matching :class:`OpenAILLM`).
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        processed_inputs = self.preprocess(inputs, **kwargs)
        if not processed_inputs:
            return []

        outputs: List[Optional[str]] = [None] * len(processed_inputs)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self._make_single_call, api_input): i
                for i, api_input in enumerate(processed_inputs)
            }
            completed = 0
            total = len(processed_inputs)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                outputs[index] = future.result()
                completed += 1
                logger.info(
                    "[%s] %d/%d calls complete", self.model_id, completed, total
                )

        return self.postprocess(outputs, **kwargs)

    # ------------------------------------------------------------------
    # Postprocess
    # ------------------------------------------------------------------
    def postprocess(self, outputs: List[Optional[str]], **kwargs) -> List[str]:
        """Strip each response; map ``None`` to empty string.

        Matches :meth:`OpenAILLM.postprocess` so the judge code path
        (``self.model.run([eval_input])[0]`` -> string) is identical.
        """
        return [output.strip() if output else "" for output in outputs]


# ---------------------------------------------------------------------------
# Model metadata registrations
# ---------------------------------------------------------------------------
#
# Registry keys carry a ``-text`` suffix because ``karma.models.gemini_asr``
# already owns the bare Google model IDs (``gemini-2.5-pro``, etc.) as ASR
# routes. ``model_path`` still pins the real Google model ID used on the wire.

GEMINI_2_5_PRO_LLM = ModelMeta(
    name="gemini-2.5-pro-text",
    model_path="gemini-2.5-pro",
    description=(
        "Google Gemini 2.5 Pro language model — primary judge API for "
        "MedCaseReasoning and related LLM-as-judge metrics."
    ),
    loader_class="karma.models.gemini_llm.GeminiLLM",
    loader_kwargs={
        "temperature": 0.0,
        "top_p": 1.0,
        "max_output_tokens": 8192,
        "max_workers": 10,
    },
    reference="https://ai.google.dev/gemini-api/docs/models",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    version="2.5",
)

GEMINI_2_5_FLASH_LLM = ModelMeta(
    name="gemini-2.5-flash-text",
    model_path="gemini-2.5-flash",
    description=(
        "Google Gemini 2.5 Flash language model — lower-latency / "
        "lower-cost alternative to 2.5 Pro for judge and direct-eval use."
    ),
    loader_class="karma.models.gemini_llm.GeminiLLM",
    loader_kwargs={
        "temperature": 0.0,
        "top_p": 1.0,
        "max_output_tokens": 8192,
        "max_workers": 10,
    },
    reference="https://ai.google.dev/gemini-api/docs/models",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    version="2.5",
)

register_model_meta(GEMINI_2_5_PRO_LLM)
register_model_meta(GEMINI_2_5_FLASH_LLM)
