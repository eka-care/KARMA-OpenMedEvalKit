import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from google.genai import types
from google import genai
from karma.models.base_model_abs import BaseModel
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.data_models.model_meta import ModelMeta, ModalityType, ModelType
from karma.registries.model_registry import register_model_meta
from karma.utils.retry import call_with_retry, is_transient_error

import logging

logger = logging.getLogger(__name__)


def _gemini_retryable(exc: BaseException) -> bool:
    return is_transient_error(exc) or "already been terminated" in str(exc)


class GeminiASR(BaseModel):
    """Gemini-based multimodal model for the KARMA framework.

    Supports both ASR (audio → text) and text generation (text → text).
    Routing is determined per-sample based on whether ``audio`` or
    ``input``/``conversation`` is populated on the DataLoaderIterable.
    """

    def __init__(
        self,
        model_name_or_path: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the Gemini service.

        Args:
            model_name_or_path: Gemini model ID to use
            api_key: Google AI API key (if None, will try to get from environment)
            thinking_budget: Optional thinking budget for enhanced reasoning
            **kwargs: Additional arguments passed to BaseModel
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            **kwargs,
        )

        self.model_id = model_name_or_path
        self.api_key = api_key or os.getenv("GOOGLE_AI_API_KEY")
        self.thinking_budget = thinking_budget

        if not self.api_key:
            raise ValueError("Google AI API key must be provided either as parameter or GOOGLE_AI_API_KEY environment variable")

        self.client = None
        self.load_model()

    def load_model(self):
        """Initialize the Google GenAI client."""
        self.client = genai.Client(api_key=self.api_key)
        self.is_loaded = True

    def _build_generation_config(self, system_prompt: Optional[str] = None):
        """Build a GenerateContentConfig honoring thinking budget and system prompt."""
        config_kwargs = {}
        if self.thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        return types.GenerateContentConfig(**config_kwargs)

    def run(self, inputs: List[DataLoaderIterable], max_workers: int = 10, **kwargs):
        """
        Run generation on the inputs in parallel.

        Each sample is dispatched to either audio transcription or text
        generation depending on which field is populated.

        Args:
            inputs: List of DataLoaderIterable objects
            max_workers: Maximum number of parallel workers (default: 10)

        Returns:
            List of generated text strings
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._dispatch_item, item): i
                for i, item in enumerate(inputs)
            }

            results = [None] * len(inputs)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = {"text": future.result(), "error": None}
                except Exception as e:
                    logger.warning("Gemini call failed for input %d: %s", index, e)
                    results[index] = {"text": "", "error": str(e)}

        return results

    def run_with_metadata(self, inputs: List[DataLoaderIterable], **kwargs):
        outputs = self.run(inputs, **kwargs)
        return [
            {
                "text": (out.get("text") or "").strip(),
                "tool_trace": "",
                "error": out.get("error"),
            }
            for out in outputs
        ]

    def _dispatch_item(self, item: DataLoaderIterable) -> str:
        """Route a single sample to the appropriate Gemini call."""
        if item.audio is not None:
            return self.transcribe(item.audio, item.input)
        return self.generate_text(item)

    def generate_text(self, item: DataLoaderIterable) -> str:
        """
        Generate text from a DataLoaderIterable using Gemini.

        Supports plain text input or multi-turn conversation, with an optional
        system prompt.
        """
        contents = []
        if item.conversation and item.conversation.conversation_turns:
            for turn in item.conversation.conversation_turns:
                role = "model" if turn.role in ("assistant", "model") else "user"
                contents.append(
                    types.Content(role=role, parts=[types.Part.from_text(text=turn.content)])
                )
        elif item.input:
            contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text=item.input)])
            )
        else:
            raise ValueError("DataLoaderIterable has neither audio nor text input")

        try:
            return call_with_retry(
                lambda: (
                    self.client.models.generate_content(
                        model=self.model_id,
                        config=self._build_generation_config(system_prompt=item.system_prompt),
                        contents=contents,
                    ).text
                    or ""
                ),
                is_retryable=_gemini_retryable,
                description="Gemini generate",
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate text with Gemini: {str(e)}") from e

    def transcribe(self, audio_bytes, instruction=None):
        """
        Transcribe audio bytes using Gemini models via the Google AI API.

        Args:
            audio_bytes (bytes): Raw audio bytes to transcribe
            instruction (str): task instruction supplied by the dataset's
                user_prompt (``item.input``). Required — this is a general model,
                so it must be told the task; it does not assume transcription.

        Returns:
            str: Transcribed text
        """
        if not instruction:
            raise ValueError(
                "GeminiASR received audio with no instruction. Set a user_prompt "
                "on the dataset (e.g. default_args={'user_prompt': 'asr_transcribe'}) "
                "or pass --dataset-args '<dataset>:user_prompt_text=...'."
            )
        # Create a temporary file to store the audio bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name

        try:
            def _call():
                # A 503 can kill the resumable upload session ("already been
                # terminated"); the SDK cannot resume it, so each attempt
                # starts a fresh upload.
                uploaded_file = self.client.files.upload(file=temp_file_path)
                response = self.client.models.generate_content(
                    model=self.model_id,
                    config=self._build_generation_config(),
                    contents=[instruction, uploaded_file],
                )
                return response.text if response.text else ""

            return call_with_retry(
                _call,
                is_retryable=_gemini_retryable,
                description="Gemini transcribe",
            )
        except Exception as e:
            raise RuntimeError(f"Failed to transcribe with Gemini: {str(e)}") from e
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def preprocess(self, inputs: List[DataLoaderIterable], **kwargs):
        """Passthrough: run() dispatches per-sample, no batch preprocess needed."""
        return inputs

    def postprocess(self, outputs: List[str], **kwargs):
        """Postprocess outputs (currently returns them as-is)."""
        return outputs


# Model metadata definitions
GeminiASR_2_0_Flash = ModelMeta(
    name="gemini-2.0-flash",
    description="Google Gemini 2.0 Flash ASR model",
    loader_class="karma.models.gemini_asr.GeminiASR",
    loader_kwargs={
        "thinking_budget": None,
    },
    revision=None,
    reference="https://ai.google.dev/gemini-api/docs",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    release_date="2024-05-14",
    version="2.0",
)

GeminiASR_2_5_Flash = ModelMeta(
    name="gemini-2.5-flash",
    description="Google Gemini 2.5 Flash ASR model",
    loader_class="karma.models.gemini_asr.GeminiASR",
    loader_kwargs={
        "thinking_budget": None,
    },
    revision=None,
    reference="https://ai.google.dev/gemini-api/docs",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    release_date="2024-05-14",
    version="2.5",
)

GeminiASR_2_5_Flash_lite = ModelMeta(
    name="gemini-2.5-flash-lite",
    description="Google Gemini 2.5 Flash Lite ASR model",
    loader_class="karma.models.gemini_asr.GeminiASR",
    loader_kwargs={
        "thinking_budget": None,
    },
    revision=None,
    reference="https://ai.google.dev/gemini-api/docs",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    release_date="2024-05-14",
    version="2.5",
)

GeminiASR_2_5_Pro = ModelMeta(
    name="gemini-2.5-pro",
    description="Google Gemini 2.5 Pro ASR model",
    loader_class="karma.models.gemini_asr.GeminiASR",
    loader_kwargs={
        "thinking_budget": None,
    },
    revision=None,
    reference="https://ai.google.dev/gemini-api/docs",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    release_date="2024-05-14",
    version="2.5",
)

GeminiASR_3_Flash_Preview = ModelMeta(
    name="gemini-3-flash-preview",
    description="Google Gemini 3 Flash Preview ASR model",
    loader_class="karma.models.gemini_asr.GeminiASR",
    loader_kwargs={
        "thinking_budget": None,
    },
    revision=None,
    reference="https://ai.google.dev/gemini-api/docs",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    release_date="2025-03-25",
    version="3",
)

GeminiASR_3_5_Flash = ModelMeta(
    name="gemini-3.5-flash",
    description="Google Gemini 3.5 Flash ASR model",
    loader_class="karma.models.gemini_asr.GeminiASR",
    loader_kwargs={
        "thinking_budget": 0,
    },
    revision=None,
    reference="https://ai.google.dev/gemini-api/docs",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    release_date="2026-06-17",
    version="3.5",
)

GeminiASR_3_6_Flash = ModelMeta(
    name="gemini-3.6-flash",
    description="Google Gemini 3.5 Flash ASR model",
    loader_class="karma.models.gemini_asr.GeminiASR",
    loader_kwargs={
        "thinking_budget": None,
    },
    revision=None,
    reference="https://ai.google.dev/gemini-api/docs",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    release_date="2026-06-17",
    version="3.5",
)

GeminiASR_3_1_Pro_Preview = ModelMeta(
    name="gemini-3.1-pro-preview",
    description="Google Gemini 3.1 Pro Preview ASR model",
    loader_class="karma.models.gemini_asr.GeminiASR",
    loader_kwargs={
        "thinking_budget": None,
    },
    revision=None,
    reference="https://ai.google.dev/gemini-api/docs",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    release_date="2026-03-09",
    version="3.1",
)

GeminiASR_3_1_Flash_Lite_Preview = ModelMeta(
    name="gemini-3.1-flash-lite-preview",
    description="Google Gemini 3.1 Flash Lite Preview ASR model",
    loader_class="karma.models.gemini_asr.GeminiASR",
    loader_kwargs={
        "thinking_budget": None,
    },
    revision=None,
    reference="https://ai.google.dev/gemini-api/docs",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    release_date="2026-03-09",
    version="3.1",
)

# Register the models
register_model_meta(GeminiASR_2_0_Flash)
register_model_meta(GeminiASR_2_5_Flash)
register_model_meta(GeminiASR_2_5_Flash_lite)
register_model_meta(GeminiASR_2_5_Pro)
register_model_meta(GeminiASR_3_Flash_Preview)
register_model_meta(GeminiASR_3_1_Pro_Preview)
register_model_meta(GeminiASR_3_5_Flash)
register_model_meta(GeminiASR_3_6_Flash)
register_model_meta(GeminiASR_3_1_Flash_Lite_Preview)