import logging
from typing import List, Optional, Any
from io import BytesIO
import torch
import librosa
from transformers import pipeline
from karma.models.base_model_abs import BaseModel
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.data_models.model_meta import ModelMeta, ModalityType, ModelType
from karma.registries.model_registry import register_model_meta

logger = logging.getLogger(__name__)


class WhisperTransformersASR(BaseModel):
    """Whisper-based ASR model using Transformers library for the KARMA framework."""

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        target_sample_rate: int = 16000,
        task: str = "transcribe",
        language: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Whisper Transformers ASR model.

        Args:
            model_name_or_path: HuggingFace model ID (e.g., "openai/whisper-base")
            device: Device to use for inference ("cuda", "cpu", or None for auto)
            target_sample_rate: Target sample rate for audio preprocessing
            task: Task type ("transcribe" or "translate")
            language: Language code for transcription (None for auto-detection)
            **kwargs: Additional model-specific parameters
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            **kwargs,
        )

        self.target_sample_rate = target_sample_rate
        self.task = task
        self.language = language
        self.pipe = None

        # Set device for pipeline
        if device is None:
            self.pipeline_device = 0 if torch.cuda.is_available() else -1
        elif device == "cuda":
            self.pipeline_device = 0
        else:
            self.pipeline_device = -1

    def load_model(self) -> None:
        """Load the Whisper model pipeline."""
        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_name_or_path,
                device=self.pipeline_device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            self.is_loaded = True
            logger.info(f"Loaded Whisper model: {self.model_name_or_path}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(f"Failed to load Whisper model: {e}") from e

    def run(
        self,
        inputs: List[DataLoaderIterable],
        **kwargs,
    ) -> List[str]:
        """
        Forward pass through the Whisper ASR model.

        Args:
            inputs: Audio inputs with DataLoaderIterable format
            **kwargs: Additional forward pass arguments (task, language, etc.)

        Returns:
            List of transcriptions
        """
        if not self.is_loaded:
            self.load_model()

        if self.pipe is None:
            raise RuntimeError("Model pipeline is not loaded")

        transcriptions = []

        for input_item in inputs:
            # Extract parameters from input or use defaults
            task = kwargs.get("task", self.task)
            language = kwargs.get("language", self.language)
            
            # Override with item-specific language if available
            if input_item.other_args and "language" in input_item.other_args:
                language = input_item.other_args["language"]

            # Preprocess audio
            processed_audio = self.preprocess(input_item)

            # Generate transcription
            generate_kwargs = {
                "repetition_penalty": kwargs.get("repetition_penalty", 1.2),
                "num_beams": kwargs.get("num_beams", 1),
                "temperature": kwargs.get("temperature", 0.01),
                "task": task,
            }
            
            # Add language if specified
            if language:
                generate_kwargs["language"] = self._map_language_code(language)

            result = self.pipe(
                processed_audio,
                generate_kwargs=generate_kwargs,
                max_new_tokens=kwargs.get("max_new_tokens", 440)
            )

            transcription = result.get("text", "")
            transcriptions.append(transcription)

        return transcriptions

    def preprocess(
        self,
        input_item: DataLoaderIterable,
        **kwargs,
    ) -> Any:
        """
        Preprocess audio input for Whisper model.

        Args:
            input_item: DataLoaderIterable item containing audio data
            **kwargs: Additional preprocessing arguments

        Returns:
            Preprocessed audio data ready for pipeline
        """
        if input_item.audio is None:
            raise ValueError("Audio data is required but not provided")

        # Load audio using librosa
        audio_data, _ = librosa.load(
            BytesIO(input_item.audio), 
            sr=self.target_sample_rate
        )
        
        return audio_data

    def postprocess(self, model_outputs: List[str], **kwargs) -> List[str]:
        """
        Postprocess model outputs into final format.

        Args:
            model_outputs: Raw transcription strings from pipeline
            **kwargs: Additional postprocessing arguments

        Returns:
            Processed transcriptions (already strings in this case)
        """
        return model_outputs

    def _map_language_code(self, language_code: str) -> str:
        """
        Map system language codes to Whisper-compatible language codes.
        
        Args:
            language_code: Input language code
            
        Returns:
            Whisper-compatible language code
        """
        # Language code mapping for Whisper
        language_mapping = {
            "en-IN": "en",
            "en-US": "en", 
            "en": "en",
            "hi": "hi",
            "ta": "ta",
            "te": "te",
            "kn": "kn",
            "ml": "ml",
            "bn": "bn",
            "gu": "gu",
            "mr": "mr",
            "pa": "pa", 
            "as": "as",
            "ur": "ur",
            "ne": "ne",
            "sa": "sa",
        }
        
        return language_mapping.get(language_code, language_code)


# Model configurations
WHISPER_LARGE_V3 = ModelMeta(
    name="openai/whisper-large-v3",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    description="OpenAI Whisper Large V3 ASR model for multilingual speech recognition",
    audio_sample_rate=16000,
    supported_audio_formats=["wav", "flac", "mp3", "m4a", "ogg"],
    loader_class="karma.models.whisper.WhisperTransformersASR",
    loader_kwargs={
        "task": "transcribe",
        "target_sample_rate": 16000,
        "language": None,  # Auto-detection
    },
    default_eval_kwargs={
        "task": "transcribe",
        "language": None,
    },
    languages=["en", "hi", "ta", "te", "kn", "ml", "bn", "gu", "mr", "pa", "as", "ur", "ne", "sa"],
    license="MIT",
    open_weights=True,
    reference="https://huggingface.co/openai/whisper-large-v3",
    release_date="2022-09-21",
    version="1.0",
    framework=["transformers", "torch"],
    n_parameters=74000000,  # ~74M parameters
)

AUDIOX_NORTH_V1 = ModelMeta(
    name="jiviai/audioX-north-v1",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    description="Jiva AI AudioX North V1 ASR model for multilingual speech recognition",
    audio_sample_rate=16000,
    supported_audio_formats=["wav", "flac", "mp3", "m4a", "ogg"],
    loader_class="karma.models.whisper.WhisperTransformersASR",
    loader_kwargs={
        "task": "transcribe",
        "target_sample_rate": 16000,
        "language": None,  # Auto-detection
    },
    default_eval_kwargs={
        "task": "transcribe",
        "language": None,
    },
    languages=["hi", "gu", "mr"],
    license="MIT",
    open_weights=True,
    reference="https://huggingface.co/jiviai/audioX-north-v1",
    release_date="2022-09-21",
    version="1.0",
    framework=["transformers", "torch"],
    n_parameters=74000000,  # ~74M parameters
)

AUDIOX_SOUTH_V1 = ModelMeta(
    name="jiviai/audioX-south-v1",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    description="Jiva AI AudioX South V1 ASR model for multilingual speech recognition",
    loader_class="karma.models.whisper.WhisperTransformersASR",
    loader_kwargs={
        "task": "transcribe",
        "target_sample_rate": 16000,
        "language": None,  # Auto-detection
    },
    default_eval_kwargs={
        "task": "transcribe",
        "language": None,
    },
    languages=["ta", "te", "kn", "ml"],
    reference="https://huggingface.co/jiviai/audioX-south-v1",
    release_date="2022-09-21",
    version="1.0",# ~74M parameters
)

register_model_meta(WHISPER_LARGE_V3)
register_model_meta(AUDIOX_NORTH_V1)
register_model_meta(AUDIOX_SOUTH_V1)