import asyncio
import base64
import json
import os
import logging
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.models.base_model_abs import BaseModel
from karma.models.mcp_client import (
    FASTMCP_AVAILABLE,
    MCPClientPool,
    discover_mcp_tools,
    mcp_tool_to_bedrock_schema,
)
from karma.data_models.model_meta import ModelMeta, ModalityType, ModelType
from karma.registries.model_registry import register_model_meta

logger = logging.getLogger(__name__)


class AWSBedrock(BaseModel):
    """AWS Bedrock-based LLM model for the KARMA framework."""

    def __init__(
        self,
        model_name_or_path: str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        max_tokens: int = 4092,
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_workers: int = 20,
        tools: Optional[List[str]] = None,
        force_tool_call: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the AWS Bedrock LLM service.

        Args:
            model_name_or_path: Bedrock model ID to use (e.g., "anthropic.claude-3-5-sonnet-20240620-v1:0")
            region_name: AWS region name (if None, will try to get from environment)
            aws_access_key_id: AWS access key ID (if None, will try to get from environment)
            aws_secret_access_key: AWS secret access key (if None, will try to get from environment)
            aws_session_token: AWS session token (if None, will try to get from environment)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Top-p sampling parameter (0.0 to 1.0)
            max_workers: Maximum number of concurrent API calls (default: 4)
            tools: Optional list of MCP server URLs for tool discovery
            **kwargs: Additional arguments passed to BaseModel
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            **kwargs,
        )

        self.model_id = model_name_or_path
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )
        self.aws_session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_workers = max_workers

        # MCP tool configuration
        self.mcp_server_urls: List[str] = tools or []
        self._bedrock_tools: Optional[List[Dict[str, Any]]] = None
        self._tool_server_map: Dict[str, str] = {}
        self.force_tool_call: Optional[str] = force_tool_call
        self.tool_trace: bool = kwargs.get("tool_trace", False)

        self.client = None
        self.load_model()

    def load_model(self):
        """Initialize the AWS Bedrock client."""
        try:
            session_kwargs = {
                "region_name": self.region_name,
            }

            if self.aws_access_key_id:
                session_kwargs["aws_access_key_id"] = self.aws_access_key_id
            if self.aws_secret_access_key:
                session_kwargs["aws_secret_access_key"] = self.aws_secret_access_key
            if self.aws_session_token:
                session_kwargs["aws_session_token"] = self.aws_session_token

            self.client = boto3.client("bedrock-runtime", **session_kwargs)
            self.is_loaded = True
            logger.info(f"AWS Bedrock client initialized with model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize AWS Bedrock client: {str(e)}")
            raise RuntimeError(
                f"Failed to initialize AWS Bedrock client: {str(e)}"
            ) from e

        if self.mcp_server_urls and self._bedrock_tools is None:
            if not FASTMCP_AVAILABLE:
                raise ImportError(
                    "The 'tools' parameter requires the fastmcp package. "
                    "Install it with: pip install 'karma-medeval[tools]'"
                )
            try:
                self._bedrock_tools, self._tool_server_map = asyncio.run(
                    discover_mcp_tools(
                        self.mcp_server_urls,
                        schema_converter=mcp_tool_to_bedrock_schema,
                    )
                )
                logger.info(
                    f"Discovered {len(self._bedrock_tools)} MCP tools from "
                    f"{len(self.mcp_server_urls)} server(s)"
                )
            except Exception as e:
                logger.error(f"Failed to discover MCP tools: {str(e)}")
                raise RuntimeError(f"MCP tool discovery failed: {str(e)}") from e

    def preprocess(
        self, inputs: List[DataLoaderIterable], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Preprocess inputs for AWS Bedrock API calls.

        Args:
            inputs: List of DataLoaderIterable objects containing text data or conversation data

        Returns:
            List of message dictionaries ready for API calls
        """
        processed_inputs = []

        for item in inputs:
            messages = []
            system_prompt = None

            # Check if conversation field exists and has data
            if item.conversation and len(item.conversation.conversation_turns) > 0:
                for turn in item.conversation.conversation_turns:
                    # Map conversation turn to Bedrock message format
                    messages.append(
                        {"role": turn.role, "content": [{"text": turn.content}]}
                    )

            # Fall back to input field if no conversation data
            elif item.input:
                messages = [{"role": "user", "content": [{"text": item.input}]}]

            # Handle system prompt
            if hasattr(item, "system_prompt") and item.system_prompt:
                system_prompt = item.system_prompt

            if item.images:
                for image in item.images:
                    # Convert image bytes to base64 for OpenAI API
                    # image_b64 = base64.b64encode(image).decode("utf-8")

                    # Add image content in OpenAI's multimodal format
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "image": {
                                        "format": "jpeg",
                                        "source": {"bytes": image},
                                    },
                                }
                            ],
                        }
                    )

            # Ensure we have at least one message
            if not messages:
                logger.warning("No input or conversation data found for item, skipping")
                continue

            api_input = {
                "modelId": self.model_id,
                "messages": messages,
                "inferenceConfig": {
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                    # "topP": self.top_p,
                },
            }

            if system_prompt:
                api_input["system"] = [{"text": system_prompt}]

            processed_inputs.append(api_input)

        return processed_inputs

    def _make_single_call(self, api_input: Dict[str, Any]) -> str:
        """
        Make a single API call to AWS Bedrock, with autonomous tool loop if MCP tools are configured.

        Args:
            api_input: Processed API input dictionary

        Returns:
            Generated text string or error message
        """
        if self._bedrock_tools:
            try:
                return asyncio.run(self._tool_loop(api_input))
            except Exception as e:
                logger.error(f"Tool loop failed for Bedrock: {str(e)}")
                return f"Error: {str(e)}"

        try:
            response = self.client.converse(**api_input)
            generated_text = response["output"]["message"]["content"][0]["text"]
            return generated_text
        except Exception as e:
            logger.error(f"Failed to generate text with AWS Bedrock: {str(e)}")
            return f"Error: {str(e)}"

    async def _tool_loop(self, api_input: Dict[str, Any]) -> str:
        """Autonomous agentic loop: call Bedrock, execute tool calls via MCP, repeat until final text.

        When ``self.tool_trace`` is enabled the return value includes the full
        conversation trace (tool calls + results + final text) so that downstream
        rubric evaluators can score retrieval behaviour.  Format mirrors
        OpenAILLM / DocAssistProtocolBot::

            [Tool Call: <name>] <args_json>
            [Tool Result: <call_id>] <content>
            <final assistant text>

        When disabled (default), only the final assistant text is returned.
        """
        messages = list(api_input["messages"])
        call_kwargs = {k: v for k, v in api_input.items() if k != "messages"}
        if self.force_tool_call:
            tool_choice = {"tool": {"name": self.force_tool_call}}
        else:
            tool_choice = {"auto": {}}
        call_kwargs["toolConfig"] = {
            "tools": self._bedrock_tools,
            "toolChoice": tool_choice,
        }

        consecutive_failures: Dict[str, int] = {}
        MAX_CONSECUTIVE_FAILURES = 3
        trace_parts: list[str] = []

        async with MCPClientPool(self._tool_server_map) as pool:
            while True:
                try:
                    response = self.client.converse(messages=messages, **call_kwargs)
                except Exception as e:
                    logger.error(f"Bedrock API call failed in tool loop: {e}")
                    raise RuntimeError(f"Bedrock API call failed: {e}") from e

                stop_reason = response.get("stopReason", "end_turn")
                output_message = response["output"]["message"]
                content_blocks = output_message.get("content", [])

                # Extract text and tool use blocks
                text_parts = []
                tool_uses = []
                for block in content_blocks:
                    if "text" in block:
                        text_parts.append(block["text"])
                    elif "toolUse" in block:
                        tool_uses.append(block["toolUse"])

                if stop_reason != "tool_use" or not tool_uses:
                    final_text = "\n".join(text_parts).strip()
                    if not self.tool_trace:
                        return final_text
                    if final_text:
                        trace_parts.append(final_text)
                    return "\n".join(trace_parts) if trace_parts else "No response"

                # Append assistant message with tool use to history
                messages.append({"role": "assistant", "content": content_blocks})

                # Execute tool calls sequentially and build tool results
                tool_result_content = []
                for tu in tool_uses:
                    tool_use_id = tu["toolUseId"]
                    tool_name = tu["name"]
                    tool_args = tu.get("input", {})
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}

                    if self.tool_trace:
                        trace_parts.append(
                            f"[Tool Call: {tool_name}] {json.dumps(tool_args, ensure_ascii=False)}"
                        )

                    try:
                        result_blocks = await pool.call_tool(tool_name, tool_args)
                        consecutive_failures.pop(tool_use_id, None)

                        # Convert structured blocks to Bedrock-native content
                        bedrock_content = []
                        for block in result_blocks:
                            if block["type"] == "text":
                                bedrock_content.append({"text": block["text"]})
                            elif block["type"] == "image":
                                img_bytes = base64.b64decode(block["data"])
                                fmt = block.get("mime_type", "image/jpeg").split("/")[
                                    -1
                                ]
                                bedrock_content.append(
                                    {
                                        "image": {
                                            "format": fmt,
                                            "source": {"bytes": img_bytes},
                                        }
                                    }
                                )
                        if not bedrock_content:
                            bedrock_content = [{"text": "Empty tool result"}]

                        tool_result_content.append(
                            {
                                "toolResult": {
                                    "toolUseId": tool_use_id,
                                    "content": bedrock_content,
                                }
                            }
                        )
                        if self.tool_trace:
                            trace_parts.append(
                                f"[Tool Result: {tool_use_id}] <tool_result consumed by LLM>"
                            )
                    except Exception as e:
                        error_msg = f"Tool execution error: {str(e)}"
                        logger.warning(
                            f"Tool '{tool_name}' (id={tool_use_id}) failed: {e}"
                        )
                        consecutive_failures[tool_use_id] = (
                            consecutive_failures.get(tool_use_id, 0) + 1
                        )
                        if (
                            consecutive_failures[tool_use_id]
                            >= MAX_CONSECUTIVE_FAILURES
                        ):
                            raise RuntimeError(
                                f"Tool '{tool_name}' failed {MAX_CONSECUTIVE_FAILURES} consecutive times. "
                                f"Last error: {error_msg}"
                            )
                        tool_result_content.append(
                            {
                                "toolResult": {
                                    "toolUseId": tool_use_id,
                                    "content": [{"text": error_msg}],
                                    "status": "error",
                                }
                            }
                        )
                        if self.tool_trace:
                            trace_parts.append(
                                f"[Tool Result: {tool_use_id}] {error_msg}"
                            )

                # Append tool results as user message
                messages.append({"role": "user", "content": tool_result_content})

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
claude_sonnet_35_v2_bedrock = ModelMeta(
    name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    description="Anthropic Claude 3.5 Sonnet via AWS Bedrock",
    loader_class="karma.models.aws_bedrock.AWSBedrock",
    loader_kwargs={
        "model_name_or_path": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "max_tokens": 4096,
        "temperature": 0.0,
        "top_p": 0.9,
    },
    revision=None,
    reference="https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-3-5-sonnet.html",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    n_parameters=None,
    memory_usage_mb=None,  # API-based, no local memory usage
    max_tokens=8192,
    embed_dim=None,
    framework=["bedrock"],
    release_date="2024-06-20",
    version="1.0",
    license=None,
    open_weights=False,
)

claude_sonnet_35_bedrock = ModelMeta(
    name="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    description="Anthropic Claude 3.5 Sonnet via AWS Bedrock",
    loader_class="karma.models.aws_bedrock.AWSBedrock",
    loader_kwargs={
        "model_name_or_path": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "max_tokens": 4096,
        "temperature": 0.0,
        "top_p": 0.9,
    },
    revision=None,
    reference="https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-3-5-sonnet.html",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    release_date="2024-06-20",
    version="1.0",
)


claude_Sonnet4_bedrock = ModelMeta(
    name="us.anthropic.claude-sonnet-4-20250514-v1:0",
    description="Anthropic Sonnet 4 via AWS Bedrock",
    loader_class="karma.models.aws_bedrock.AWSBedrock",
    loader_kwargs={
        "model_name_or_path": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "max_tokens": 8192,
        "temperature": 0.0,
        "top_p": 0.9,
    },
    revision=None,
    reference="https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-3-haiku.html",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    release_date="2024-03-07",
    version="1.0",
)

claude_Sonnet45_bedrock = ModelMeta(
    name="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    description="Anthropic Sonnet 4.5 via AWS Bedrock",
    loader_class="karma.models.aws_bedrock.AWSBedrock",
    loader_kwargs={
        "model_name_or_path": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "max_tokens": 8192,
        "temperature": 0.0,
    },
    revision=None,
    reference="https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-3-haiku.html",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    release_date="2024-03-07",
    version="1.0",
)

claude_Sonnet46_bedrock = ModelMeta(
    name="us.anthropic.claude-sonnet-4-6",
    description="Anthropic Sonnet 4.5 via AWS Bedrock",
    loader_class="karma.models.aws_bedrock.AWSBedrock",
    loader_kwargs={
        "model_name_or_path": "us.anthropic.claude-sonnet-4-6",
        "max_tokens": 8192,
        "temperature": 0.0,
    },
    revision=None,
    reference="https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-3-haiku.html",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    release_date="2024-03-07",
    version="1.0",
)


# Register the models
register_model_meta(claude_sonnet_35_bedrock)
register_model_meta(claude_sonnet_35_v2_bedrock)
register_model_meta(claude_Sonnet4_bedrock)
register_model_meta(claude_Sonnet45_bedrock)
register_model_meta(claude_Sonnet46_bedrock)
