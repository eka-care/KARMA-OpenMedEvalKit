import os
import logging
import base64
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from karma.models.base_model_abs import BaseModel
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
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_workers: int = 4,
        **kwargs,
    ):
        """
        Initialize the OpenAI LLM service.

        Args:
            model_name_or_path: OpenAI model ID to use (e.g., "gpt-4o", "gpt-4o-mini")
            api_key: OpenAI API key (if None, will try to get from environment)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Top-p sampling parameter (0.0 to 1.0)
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            max_workers: Maximum number of concurrent API calls (default: 4)
            **kwargs: Additional arguments passed to BaseModel
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            **kwargs,
        )

        self.model_id = model_name_or_path
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.max_workers = max_workers

        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided either as parameter or OPENAI_API_KEY environment variable"
            )

        self.client: OpenAI = None
        self.load_model()

    def load_model(self, **kwargs) -> None:
        """Initialize the OpenAI client."""
        try:
            self.client = OpenAI(api_key=self.api_key)
            self.is_loaded = True
            logger.info(f"OpenAI LLM client initialized with model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}") from e

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
            if self.model_id != "o3":
                # o3 does not accept these keys
                message_dict.update(
                    {
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "frequency_penalty": self.frequency_penalty,
                        "presence_penalty": self.presence_penalty,
                    }
                )
            processed_inputs.append(message_dict)

        return processed_inputs

    def _make_single_call(self, api_input: Dict[str, Any]) -> str:
        """
        Make a single API call to OpenAI.

        Args:
            api_input: Processed API input dictionary

        Returns:
            Generated text string or error message
        """
        try:
            response = self.client.chat.completions.create(**api_input)
            # Extract the generated text
            generated_text = response.choices[0].message.content
            return generated_text
        except Exception as e:
            logger.error(f"Failed to generate text with OpenAI: {str(e)}")
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
    reference="https://platform.openai.com/docs/models/gpt-3-5-turbo",
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


# Register the models
register_model_meta(GPT4o_LLM)
register_model_meta(GPT4o_Mini_LLM)
register_model_meta(GPT35_Turbo_LLM)
register_model_meta(GPT41_LLM)
register_model_meta(GPTo3_LLM)
