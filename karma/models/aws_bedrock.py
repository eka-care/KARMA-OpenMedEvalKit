import os
import logging
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.models.base_model_abs import BaseModel
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
        max_workers: int = 4,
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
                                        "format": 'jpeg',
                                        "source": {
                                            "bytes": image
                                        }
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
                    "topP": self.top_p,
                },
            }

            if system_prompt:
                api_input["system"] = [{"text": system_prompt}]

            processed_inputs.append(api_input)

        return processed_inputs

    def _make_single_call(self, api_input: Dict[str, Any]) -> str:
        """
        Make a single API call to AWS Bedrock.

        Args:
            api_input: Processed API input dictionary

        Returns:
            Generated text string or error message
        """
        try:
            response = self.client.converse(**api_input)
            # Extract the generated text from the response
            generated_text = response["output"]["message"]["content"][0]["text"]
            return generated_text
        except Exception as e:
            logger.error(f"Failed to generate text with AWS Bedrock: {str(e)}")
            return f"Error: {str(e)}"

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

# Register the models
register_model_meta(claude_sonnet_35_bedrock)
register_model_meta(claude_sonnet_35_v2_bedrock)
register_model_meta(claude_Sonnet4_bedrock)
