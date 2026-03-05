import asyncio
import json
import os
import logging
import base64
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from karma.models.base_model_abs import BaseModel
from karma.models.mcp_client import FASTMCP_AVAILABLE, MCPClientPool, discover_mcp_tools
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.data_models.model_meta import ModelMeta, ModalityType, ModelType
from karma.registries.model_registry import register_model_meta

logger = logging.getLogger(__name__)

# the default system prompt for openai models as per
# https://github.com/openai/simple-evals/blob/main/sampler/chat_completion_sampler.py#L9


class OpenAILLM(BaseModel):
    """OpenAI-based LLM model for the KARMA framework."""

    def __init__(
        self,
        model_name_or_path: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_workers: int = 20,
        tools: Optional[List[str]] = None,
        force_tool_call: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the OpenAI LLM service.

        Args:
            model_name_or_path: OpenAI model ID to use (e.g., "gpt-4o", "gpt-4o-mini")
            api_key: OpenAI API key (if None, will try to get from environment)
            max_completion_tokens: Maximum tokens to generate for GPT-5 and newer APIs
            max_tokens: Backwards compatible alias for previous callers (deprecated)
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Top-p sampling parameter (0.0 to 1.0)
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            max_workers: Maximum number of concurrent API calls (default: 4)
            force_tool_call: Optional tool name to force the model to call (sets tool_choice to "required" with the specific function)
            **kwargs: Additional arguments passed to BaseModel
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            **kwargs,
        )

        self.model_id = model_name_or_path
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # gpt-5 family only supports temperature=1
        if "gpt-5" in model_name_or_path.lower() and temperature != 1.0:
            logger.info(
                f"Model {model_name_or_path} only supports temperature=1. "
                f"Overriding temperature={temperature} -> 1.0"
            )
            temperature = 1.0

        if max_completion_tokens is not None and max_tokens is not None:
            if max_completion_tokens != max_tokens:
                logger.warning(
                    "Both max_completion_tokens (%s) and max_tokens (%s) provided; "
                    "using max_completion_tokens",
                    max_completion_tokens,
                    max_tokens,
                )
        if max_completion_tokens is None:
            max_completion_tokens = max_tokens
        if max_completion_tokens is None:
            max_completion_tokens = 4096

        self.max_completion_tokens = max_completion_tokens
        self.max_tokens = self.max_completion_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.max_workers = max_workers
        # Optional parameter to request model reasoning effort (e.g., 'low', 'medium', 'high')
        self.reasoning_effort = kwargs.get("reasoning_effort", None)

        # MCP tool configuration
        self.mcp_server_urls: List[str] = tools or []
        self._openai_tools: Optional[List[Dict[str, Any]]] = None
        self._tool_server_map: Dict[str, str] = {}
        self.force_tool_call: Optional[str] = force_tool_call
        self.tool_trace: bool = kwargs.get("tool_trace", False)

        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided either as parameter or OPENAI_API_KEY environment variable"
            )

        self.client: OpenAI = None
        self.load_model()

    def load_model(self, **kwargs) -> None:
        """Initialize the OpenAI client and optionally discover MCP tools."""
        try:
            self.client = OpenAI(api_key=self.api_key)
            self.is_loaded = True
            logger.info(f"OpenAI LLM client initialized with model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}") from e

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
                    f"Discovered {len(self._openai_tools)} MCP tools from "
                    f"{len(self.mcp_server_urls)} server(s)"
                )
            except Exception as e:
                logger.error(f"Failed to discover MCP tools: {str(e)}")
                raise RuntimeError(f"MCP tool discovery failed: {str(e)}") from e

    def preprocess(
        self, inputs: List[DataLoaderIterable], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Preprocess inputs for OpenAI API calls.

        Args:
            inputs: List of DataLoaderIterable objects containing text data or conversation data

        Returns:
            List of message dictionaries ready for API calls
        """
        processed_inputs = []
        for item in inputs:
            messages = []

            # Check if conversation field exists and has data
            if item.conversation:
                if len(item.conversation.conversation_turns) > 0:
                    for turn in item.conversation.conversation_turns:
                        # Map conversation turn to OpenAI message format
                        messages.append({"role": turn.role, "content": turn.content})

            # Fall back to input field if no conversation data
            elif item.input:
                messages = [{"role": "user", "content": item.input}]

            # Add system prompt if available (for backward compatibility)
            if hasattr(item, "system_prompt") and item.system_prompt:
                # Insert system message at the beginning if not already present
                if not messages or messages[0]["role"] != "system":
                    messages.insert(
                        0, {"role": "developer", "content": item.system_prompt}
                    )
            if item.images:
                for image in item.images:
                    # Convert image bytes to base64 for OpenAI API
                    image_b64 = base64.b64encode(image).decode("utf-8")

                    # Add image content in OpenAI's multimodal format
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}"
                                    },
                                }
                            ],
                        }
                    )

            # Ensure we have at least one message
            if not messages:
                logger.warning("No input or conversation data found for item, skipping")
                continue

            message_dict = {
                "messages": messages,
                "model": self.model_id,
            }
            _no_sampling_params = {"o3"} | {
                m for m in [self.model_id] if m.startswith("gpt-5")
            }
            if self.model_id not in _no_sampling_params:
                # Some models (o3, gpt-5 family) do not accept sampling/penalty params
                message_dict.update(
                    {
                        "max_completion_tokens": self.max_completion_tokens,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "frequency_penalty": self.frequency_penalty,
                        "presence_penalty": self.presence_penalty,
                    }
                )
            else:
                message_dict["max_completion_tokens"] = self.max_completion_tokens
            if self.reasoning_effort is not None:
                # forward reasoning preference to OpenAI API if provided
                message_dict["reasoning_effort"] = self.reasoning_effort
            processed_inputs.append(message_dict)

        return processed_inputs

    def _make_single_call(self, api_input: Dict[str, Any]) -> str:
        """
        Make a single API call to OpenAI, with autonomous tool loop if MCP tools are configured.

        Args:
            api_input: Processed API input dictionary

        Returns:
            Generated text string or error message
        """
        if self._openai_tools:
            try:
                return asyncio.run(self._tool_loop(api_input))
            except Exception as e:
                logger.error(f"Tool loop failed for OpenAI: {str(e)}")
                return f"Error: {str(e)}"

        try:
            response = self.client.chat.completions.create(**api_input)
            generated_text = response.choices[0].message.content
            return generated_text
        except Exception as e:
            logger.error(f"Failed to generate text with OpenAI: {str(e)}")
            return f"Error: {str(e)}"

    async def _tool_loop(self, api_input: Dict[str, Any]) -> str:
        """Autonomous agentic loop: call LLM, execute tool calls via MCP, repeat until final text.

        When ``self.tool_trace`` is enabled the return value includes the full
        conversation trace (tool calls + results + final text) so that downstream
        rubric evaluators can score retrieval behaviour.  Format mirrors
        DocAssistProtocolBot::

            [Tool Call: <name>] <args_json>
            [Tool Result: <call_id>] <content>
            <final assistant text>

        When disabled (default), only the final assistant text is returned.
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
                    response = self.client.chat.completions.create(
                        messages=messages, **call_kwargs
                    )
                except Exception as e:
                    logger.error(f"OpenAI API call failed in tool loop: {e}")
                    raise RuntimeError(f"OpenAI API call failed: {e}") from e

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

                        # Check if there are any image blocks
                        has_images = any(b["type"] == "image" for b in result_blocks)

                        if has_images:
                            # Build multi-part content for OpenAI
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
                            # Text-only: simple string for backward compatibility
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
                        error_msg = f"Tool execution error: {str(e)}"
                        logger.warning(
                            f"Tool '{tool_name}' (call_id={tool_call_id}) failed: {e}"
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

    def run(self, inputs: List[DataLoaderIterable], **kwargs) -> List[str]:
        """
        Run text generation on the input prompts in parallel.

        Args:
            inputs: List of DataLoaderIterable objects containing text data

        Returns:
            List of generated text strings
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        processed_inputs = self.preprocess(inputs, **kwargs)

        # Handle empty inputs
        if not processed_inputs:
            return []

        # Use ThreadPoolExecutor for parallel API calls
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all API calls and track their order
            future_to_index = {
                executor.submit(self._make_single_call, api_input): i
                for i, api_input in enumerate(processed_inputs)
            }

            # Initialize results list with correct size
            outputs = [None] * len(processed_inputs)

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                result = future.result()
                outputs[index] = result

        return self.postprocess(outputs, **kwargs)

    def postprocess(self, outputs: List[str], **kwargs) -> List[str]:
        """
        Postprocess model outputs.

        Args:
            outputs: List of generated text strings

        Returns:
            List of processed outputs
        """
        return [output.strip() if output else "" for output in outputs]


# Model metadata definitions
GPT4o_LLM = ModelMeta(
    name="gpt-4o",
    description="OpenAI GPT-4o language model",
    loader_class="karma.models.openai_llm.OpenAILLM",
    loader_kwargs={
        "model_name_or_path": "gpt-4o",
        "max_tokens": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    reference="https://platform.openai.com/docs/models/gpt-4o",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    release_date="2024-05-13",
    version="1.0",
)

GPT4o_Mini_LLM = ModelMeta(
    name="gpt-4o-mini",
    description="OpenAI GPT-4o Mini language model",
    loader_class="karma.models.openai_llm.OpenAILLM",
    loader_kwargs={
        "model_name_or_path": "gpt-4o-mini",
        "max_tokens": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    revision=None,
    reference="https://platform.openai.com/docs/models/gpt-4o-mini",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    release_date="2024-07-18",
    version="1.0",
)

GPT35_Turbo_LLM = ModelMeta(
    name="gpt-3.5-turbo",
    description="OpenAI GPT-3.5 Turbo language model",
    loader_class="karma.models.openai_llm.OpenAILLM",
    loader_kwargs={
        "model_name_or_path": "gpt-3.5-turbo",
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    revision=None,
    reference="https://platform.openai.com/docs/models/gpt-3-5-turbo",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    release_date="2023-03-01",
    version="1.0",
)


GPT41_LLM = ModelMeta(
    name="gpt-4.1",
    description="OpenAI GPT-3.5 Turbo language model",
    loader_class="karma.models.openai_llm.OpenAILLM",
    loader_kwargs={
        "model_name_or_path": "gpt-4.1-2025-04-14",
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    revision=None,
    reference="https://platform.openai.com/docs/models/gpt-4.1",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    release_date="2025-04-14",
    version="1.0",
)

GPT41_MINI_LLM = ModelMeta(
    name="gpt-4.1-mini",
    description="OpenAI GPT-4.1 Mini language model",
    loader_class="karma.models.openai_llm.OpenAILLM",
    loader_kwargs={
        "model_name_or_path": "gpt-4.1-mini-2025-04-14",
        "max_tokens": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    revision=None,
    reference="https://platform.openai.com/docs/models/gpt-4.1-mini",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    release_date="2025-04-14",
    version="1.0",
)

GPTo3_LLM = ModelMeta(
    name="o3",
    description="OpenAI GPT-4o language model",
    loader_class="karma.models.openai_llm.OpenAILLM",
    loader_kwargs={
        "model_name_or_path": "o3",
        "max_tokens": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    reference="https://platform.openai.com/docs/models/o3",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    release_date="2025-04-16",
    version="1.0",
)

GPT5_LLM = ModelMeta(
    name="gpt-5",
    description="OpenAI GPT-5 language model",
    loader_class="karma.models.openai_llm.OpenAILLM",
    loader_kwargs={
        "model_name_or_path": "gpt-5",
        "max_completion_tokens": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    reference="https://platform.openai.com/docs/models/gpt-5",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    release_date="2025-08-07",
    version="1.0",
)

GPT5_NANO_LLM = ModelMeta(
    name="gpt-5-nano",
    description="OpenAI GPT-5-nano language model",
    loader_class="karma.models.openai_llm.OpenAILLM",
    loader_kwargs={
        "model_name_or_path": "gpt-5-nano",
        "max_completion_tokens": 4096,
        "temperature": 1.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    reference="https://platform.openai.com/docs/models/gpt-5",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    release_date="2025-08-07",
    version="1.0",
)

GPT5_MINI_LLM = ModelMeta(
    name="gpt-5-mini",
    description="OpenAI GPT-5-mini language model",
    loader_class="karma.models.openai_llm.OpenAILLM",
    loader_kwargs={
        "model_name_or_path": "gpt-5-mini",
        "max_completion_tokens": 4096,
        "temperature": 1.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    reference="https://platform.openai.com/docs/models/gpt-5",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    release_date="2025-08-07",
    version="1.0",
)

GPT52_LLM = ModelMeta(
    name="gpt-5.2",
    description="OpenAI GPT-5.2 language model",
    loader_class="karma.models.openai_llm.OpenAILLM",
    loader_kwargs={
        "model_name_or_path": "gpt-5.2",
        "max_completion_tokens": 4096,
        "temperature": 1.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "reasoning_effort": "medium",
    },
    reference="https://platform.openai.com/docs/models/gpt-5.2",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    release_date="2025-12-11",
    version="1.0",
)


# Register the models
register_model_meta(GPT4o_LLM)
register_model_meta(GPT4o_Mini_LLM)
register_model_meta(GPT35_Turbo_LLM)
register_model_meta(GPT41_LLM)
register_model_meta(GPT41_MINI_LLM)
register_model_meta(GPTo3_LLM)
register_model_meta(GPT5_LLM)
register_model_meta(GPT5_NANO_LLM)
register_model_meta(GPT5_MINI_LLM)
register_model_meta(GPT52_LLM)
