"""Sarvam AI LLM adapter for KARMA.

Subclasses :class:`karma.models.openai_llm.OpenAILLM` so we reuse the MCP
tool-calling loop, ``run`` orchestration, and post-processing from the
OpenAI path while routing requests to Sarvam's OpenAI-compatible endpoint
(``https://api.sarvam.ai/v1``).

Key deltas vs. the parent class
-------------------------------
* Credentials: uses ``SARVAM_API_KEY`` (not ``OPENAI_API_KEY``).
* Base URL: the OpenAI client is re-instantiated against Sarvam's host.
* Rate limits: default ``max_workers=1`` to stay well below the
  60 req/min free-plan ceiling.
* Reasoning mode: Sarvam is always in reasoning mode — responses come back
  as ``content=None`` when ``max_tokens`` is too low.  We default
  ``max_tokens=16384`` and wrap every API call with exponential backoff
  that retries on empty / None / truncated-at-length responses.
* Request schema rewrites (performed in :meth:`preprocess`):
    1. ``max_completion_tokens`` → ``max_tokens`` (Sarvam 400s / returns
       silent empty content on the OpenAI-native key).
    2. ``role="developer"`` → ``role="system"`` (Sarvam 400s on the
       ``developer`` role that the parent class emits by default).
    3. Zero-valued ``frequency_penalty`` / ``presence_penalty`` stripped
       because some Sarvam tool-loop paths 400 on them.
* Retry wrapper :meth:`_call_api_with_retry` covers BOTH the non-tool
  path (via :meth:`_make_single_call`) and the tool-loop path (via the
  overridden :meth:`_tool_loop`).
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.data_models.model_meta import ModalityType, ModelMeta, ModelType
from karma.models.mcp_client import (
    FASTMCP_AVAILABLE,
    MCPClientPool,
    discover_mcp_tools,
)
from karma.models.openai_llm import OpenAILLM
from karma.registries.model_registry import register_model_meta

logger = logging.getLogger(__name__)


class SarvamLLM(OpenAILLM):
    """Sarvam AI LLM using the OpenAI-compatible API."""

    # Exponential-backoff knobs for silent empty-content responses
    _RETRY_MAX_ATTEMPTS = 3
    _RETRY_INITIAL_DELAY = 30
    _RETRY_BACKOFF_FACTOR = 2

    def __init__(
        self,
        model_name_or_path: str = "sarvam-105b",
        api_key: Optional[str] = None,
        base_url: str = "https://api.sarvam.ai/v1",
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_workers: int = 1,
        tools: Optional[List[str]] = None,
        force_tool_call: Optional[str] = None,
        **kwargs,
    ):
        """Initialise the Sarvam adapter.

        Args:
            model_name_or_path: Sarvam model ID, e.g. ``"sarvam-105b"`` or
                ``"sarvam-30b"``.
            api_key: Sarvam API key; falls back to ``SARVAM_API_KEY``.
            base_url: Sarvam OpenAI-compatible endpoint.
            max_completion_tokens: Honoured for parity with the parent
                class; ``preprocess`` rewrites it to ``max_tokens`` in the
                outgoing request body.
            max_tokens: Sarvam runs in reasoning mode, so responses with a
                low cap silently return empty content. Default 4096 — this
                is the starter-tier subscription ceiling for both
                ``sarvam-105b`` and ``sarvam-30b``. Upgrade the tier to
                request higher values (e.g. 16384); the value is capped
                at the tier limit irrespective of this default.
            temperature/top_p/frequency_penalty/presence_penalty:
                Standard OpenAI-compatible sampling params.
            max_workers: Concurrent API calls. 60 rpm free-plan limit
                means 1 is the only safe default.
            tools: Optional list of MCP server URLs.
            force_tool_call: Optional tool name to force.
            **kwargs: Forwarded to :class:`OpenAILLM` / :class:`BaseModel`.
        """
        # Resolve the API key early. We keep a local copy for the warning
        # below; the parent __init__ will still read from kwargs via the
        # shared OPENAI_API_KEY fallback, which is why we explicitly pass
        # the resolved value through.
        resolved_api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not resolved_api_key:
            logger.warning(
                "SARVAM_API_KEY is not set — SarvamLLM initialisation will "
                "fail when the parent class validates credentials."
            )

        # Store base_url BEFORE super().__init__(): the parent's __init__
        # calls self.load_model() which dispatches through MRO into
        # SarvamLLM.load_model(), and that override reads self.base_url.
        self.base_url = base_url

        # Track the effective max_tokens value so preprocess() can use it
        # as a fallback when rewriting the request body.
        self._sarvam_max_tokens = max_tokens or max_completion_tokens or 4096

        super().__init__(
            model_name_or_path=model_name_or_path,
            api_key=resolved_api_key,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_workers=max_workers,
            tools=tools,
            force_tool_call=force_tool_call,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Client + MCP tool bootstrap
    # ------------------------------------------------------------------
    def load_model(self, **kwargs) -> None:
        """Instantiate the OpenAI client against Sarvam's base URL.

        We do NOT call ``super().load_model()`` because it instantiates
        the client against OpenAI's default host. The MCP discovery block
        mirrors the parent's logic exactly (see
        ``OpenAILLM.load_model:126-128``).
        """
        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.is_loaded = True
            logger.info(
                "Sarvam LLM client initialised with model=%s base_url=%s",
                self.model_id,
                self.base_url,
            )
        except Exception as e:
            logger.error("Failed to initialise Sarvam client: %s", e)
            raise RuntimeError(f"Failed to initialise Sarvam client: {e}") from e

        if self.mcp_server_urls and self._openai_tools is None:
            if not FASTMCP_AVAILABLE:
                raise ImportError(
                    "The 'tools' parameter requires the fastmcp package. "
                    "Install it with: pip install 'karma-medeval[tools]'"
                )
            try:
                self._openai_tools, self._tool_server_map = asyncio.run(
                    discover_mcp_tools(self.mcp_server_urls)
                )
                logger.info(
                    "Discovered %d MCP tools from %d server(s)",
                    len(self._openai_tools),
                    len(self.mcp_server_urls),
                )
            except Exception as e:
                logger.error("Failed to discover MCP tools: %s", e)
                raise RuntimeError(f"MCP tool discovery failed: {e}") from e

    # ------------------------------------------------------------------
    # Request-body rewrite
    # ------------------------------------------------------------------
    def preprocess(
        self, inputs: List[DataLoaderIterable], **kwargs
    ) -> List[Dict[str, Any]]:
        """Build OpenAI-shaped message dicts, then rewrite for Sarvam.

        Three in-place transforms on every per-sample message dict:

        (a) Key rename — ``max_completion_tokens`` → ``max_tokens``.
            Sarvam silently returns ``content=None`` on the OpenAI-native
            key; this is the single most important transform.
        (b) Role conversion — ``role="developer"`` → ``role="system"``.
            The parent class inserts system prompts with ``role=developer``
            (``openai_llm.py:169``), which Sarvam 400s on.
        (c) Defensive zero-penalty strip — drop ``frequency_penalty`` /
            ``presence_penalty`` when they equal 0.0; some Sarvam
            tool-loop paths 400 on the zero values.
        """
        processed_inputs = super().preprocess(inputs, **kwargs)

        for message_dict in processed_inputs:
            # (a) Key rename — critical. Pop first, then assign, so the
            # key is only ever present under its Sarvam name.
            message_dict["max_tokens"] = message_dict.pop(
                "max_completion_tokens", self._sarvam_max_tokens
            )
            logger.debug(
                "Sarvam outgoing max_tokens=%s (model=%s)",
                message_dict["max_tokens"],
                self.model_id,
            )

            # (b) Role conversion: developer -> system
            for message in message_dict.get("messages", []):
                if message.get("role") == "developer":
                    message["role"] = "system"

            # (c) Defensive zero-penalty strip
            if message_dict.get("frequency_penalty") == 0.0:
                message_dict.pop("frequency_penalty", None)
            if message_dict.get("presence_penalty") == 0.0:
                message_dict.pop("presence_penalty", None)

        return processed_inputs

    # ------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------
    def _call_api_with_retry(self, **call_kwargs):
        """Wrap ``client.chat.completions.create`` with exponential backoff.

        Retry triggers:

        * ``response.choices[0].message.content is None``
        * ``response.choices[0].message.content == ""``
        * ``finish_reason == "length"`` with empty / None content
          (reasoning-mode truncation)

        We retry up to :attr:`_RETRY_MAX_ATTEMPTS` times with a 30s
        initial delay doubled each attempt. If the budget is exhausted we
        return the last response unchanged so the caller's existing
        empty-handling stays in charge.
        """
        delay = self._RETRY_INITIAL_DELAY
        last_response = None

        for attempt in range(1, self._RETRY_MAX_ATTEMPTS + 1):
            response = self.client.chat.completions.create(**call_kwargs)
            last_response = response

            choice = response.choices[0]
            content = choice.message.content
            finish_reason = getattr(choice, "finish_reason", None)

            is_empty_content = content is None or (
                isinstance(content, str) and content.strip() == ""
            )
            length_truncated_empty = (
                finish_reason == "length" and is_empty_content
            )

            # Accept: non-empty content OR finish_reason indicates the
            # model made a tool call (content is legitimately empty in
            # that case).
            if (not is_empty_content) or finish_reason == "tool_calls" or (
                choice.message.tool_calls
            ):
                return response

            trigger = (
                "length-truncated empty content"
                if length_truncated_empty
                else ("None content" if content is None else "empty content")
            )

            if attempt >= self._RETRY_MAX_ATTEMPTS:
                logger.error(
                    "Sarvam retry budget exhausted after %d attempts "
                    "(model=%s, last trigger=%s); returning last response.",
                    attempt,
                    self.model_id,
                    trigger,
                )
                return response

            logger.warning(
                "Sarvam empty-content retry %d/%d (model=%s, trigger=%s); "
                "sleeping %ds then retrying.",
                attempt,
                self._RETRY_MAX_ATTEMPTS - 1,
                self.model_id,
                trigger,
                delay,
            )
            time.sleep(delay)
            delay *= self._RETRY_BACKOFF_FACTOR

        # Defensive fallback (unreachable under normal flow — the loop
        # above either returns a live response or the last response when
        # the retry budget is exhausted).
        return last_response

    # ------------------------------------------------------------------
    # Single-call override (non-tool path)
    # ------------------------------------------------------------------
    def _make_single_call(self, api_input: Dict[str, Any]) -> str:
        """Sarvam-aware single-call path.

        * Tool path: delegate to the overridden :meth:`_tool_loop` (same
          routing the parent does).
        * Non-tool path: wrap the ``create`` call in
          :meth:`_call_api_with_retry` to cover Sarvam's silent empty
          responses.
        """
        if self._openai_tools:
            try:
                return asyncio.run(self._tool_loop(api_input))
            except Exception as e:
                logger.error("Tool loop failed for Sarvam: %s", e)
                return f"Error: {e}"

        try:
            response = self._call_api_with_retry(**api_input)
            generated_text = response.choices[0].message.content
            return generated_text
        except Exception as e:
            logger.error("Failed to generate text with Sarvam: %s", e)
            return f"Error: {e}"

    # ------------------------------------------------------------------
    # Tool-loop override (wraps the inline create call)
    # ------------------------------------------------------------------
    async def _tool_loop(self, api_input: Dict[str, Any]) -> str:
        """Mirror of :meth:`OpenAILLM._tool_loop` with one change.

        The only modification vs. the parent is that the inline
        ``self.client.chat.completions.create(...)`` at
        ``openai_llm.py:280-282`` is routed through
        :meth:`_call_api_with_retry` so Sarvam's silent-empty retries
        cover the agentic path as well as the plain path.
        All tool dispatch, finish_reason handling, trace_parts
        accumulation, and failure-count logic are preserved verbatim.
        """
        messages = list(api_input["messages"])
        call_kwargs = {k: v for k, v in api_input.items() if k != "messages"}
        call_kwargs["tools"] = self._openai_tools
        if self.force_tool_call:
            call_kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": self.force_tool_call},
            }
        else:
            call_kwargs["tool_choice"] = "auto"

        consecutive_failures: Dict[str, int] = {}
        MAX_CONSECUTIVE_FAILURES = 3
        trace_parts: list[str] = []

        async with MCPClientPool(self._tool_server_map) as pool:
            while True:
                try:
                    # Sarvam-specific: retry-wrapped create.
                    response = self._call_api_with_retry(
                        messages=messages, **call_kwargs
                    )
                except Exception as e:
                    logger.error("Sarvam API call failed in tool loop: %s", e)
                    raise RuntimeError(f"Sarvam API call failed: {e}") from e

                choice = response.choices[0]
                assistant_message = choice.message

                if choice.finish_reason == "stop" or not assistant_message.tool_calls:
                    final_text = (assistant_message.content or "").strip()
                    if not self.tool_trace:
                        return final_text
                    if final_text:
                        trace_parts.append(final_text)
                    return "\n".join(trace_parts) if trace_parts else "No response"

                messages.append(assistant_message.model_dump(exclude_unset=True))

                for tc in assistant_message.tool_calls:
                    tool_call_id = tc.id
                    tool_name = tc.function.name

                    try:
                        tool_args = (
                            json.loads(tc.function.arguments)
                            if tc.function.arguments
                            else {}
                        )
                    except json.JSONDecodeError as e:
                        error_content = f"Error: could not parse tool arguments: {e}"
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": error_content,
                            }
                        )
                        if self.tool_trace:
                            trace_parts.append(
                                f"[Tool Call: {tool_name}] {tc.function.arguments}"
                            )
                            trace_parts.append(
                                f"[Tool Result: {tool_call_id}] {error_content}"
                            )
                        consecutive_failures[tool_call_id] = (
                            consecutive_failures.get(tool_call_id, 0) + 1
                        )
                        if (
                            consecutive_failures[tool_call_id]
                            >= MAX_CONSECUTIVE_FAILURES
                        ):
                            raise RuntimeError(
                                f"Tool '{tool_name}' failed {MAX_CONSECUTIVE_FAILURES} consecutive times. "
                                f"Last error: {error_content}"
                            )
                        continue

                    if self.tool_trace:
                        trace_parts.append(
                            f"[Tool Call: {tool_name}] {json.dumps(tool_args, ensure_ascii=False)}"
                        )

                    try:
                        result_blocks = await pool.call_tool(tool_name, tool_args)
                        consecutive_failures.pop(tool_call_id, None)

                        has_images = any(b["type"] == "image" for b in result_blocks)

                        if has_images:
                            openai_content = []
                            for block in result_blocks:
                                if block["type"] == "text":
                                    openai_content.append(
                                        {"type": "text", "text": block["text"]}
                                    )
                                elif block["type"] == "image":
                                    mime = block.get("mime_type", "image/jpeg")
                                    openai_content.append(
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:{mime};base64,{block['data']}"
                                            },
                                        }
                                    )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": openai_content,
                                }
                            )
                        else:
                            result_str = "\n".join(
                                b["text"] for b in result_blocks if b["type"] == "text"
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": result_str,
                                }
                            )

                        if self.tool_trace:
                            trace_parts.append(
                                f"[Tool Result: {tool_call_id}] <tool_result consumed by LLM>"
                            )
                    except Exception as e:
                        error_msg = f"Tool execution error: {e}"
                        logger.warning(
                            "Tool '%s' (call_id=%s) failed: %s",
                            tool_name,
                            tool_call_id,
                            e,
                        )
                        consecutive_failures[tool_call_id] = (
                            consecutive_failures.get(tool_call_id, 0) + 1
                        )
                        if (
                            consecutive_failures[tool_call_id]
                            >= MAX_CONSECUTIVE_FAILURES
                        ):
                            raise RuntimeError(
                                f"Tool '{tool_name}' failed {MAX_CONSECUTIVE_FAILURES} consecutive times. "
                                f"Last error: {error_msg}"
                            )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": error_msg,
                            }
                        )
                        if self.tool_trace:
                            trace_parts.append(
                                f"[Tool Result: {tool_call_id}] {error_msg}"
                            )


# ---------------------------------------------------------------------------
# Model metadata registrations
# ---------------------------------------------------------------------------

SARVAM_105B_LLM = ModelMeta(
    name="sarvam-105b",
    description="Sarvam AI 105B MoE language model (OpenAI-compatible API).",
    loader_class="karma.models.sarvam_llm.SarvamLLM",
    loader_kwargs={
        "model_name_or_path": "sarvam-105b",
        # 4096 = starter-tier ceiling. Bump to 16384 on higher tiers.
        "max_tokens": 4096,
        "max_workers": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    reference="https://www.sarvam.ai/",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    languages=["eng-Latn", "hin-Deva"],
)

SARVAM_30B_LLM = ModelMeta(
    name="sarvam-30b",
    description="Sarvam AI 30B MoE language model (OpenAI-compatible API).",
    loader_class="karma.models.sarvam_llm.SarvamLLM",
    loader_kwargs={
        "model_name_or_path": "sarvam-30b",
        # 4096 = starter-tier ceiling. Bump to 16384 on higher tiers.
        "max_tokens": 4096,
        "max_workers": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    reference="https://www.sarvam.ai/",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    languages=["eng-Latn", "hin-Deva"],
)

register_model_meta(SARVAM_105B_LLM)
register_model_meta(SARVAM_30B_LLM)
