import logging
from typing import List, Dict, Union, Any
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from karma.models.base_model_abs import BaseHFModel
from karma.models.model_meta import ModelMeta, ModelType, ModalityType
from karma.registries.model_registry import register_model_meta

logger = logging.getLogger(__name__)


class IndicConformerASR(BaseHFModel):
    """Indic Conformer ASR model for multilingual speech recognition."""

    def __init__(
        self,
        model_name_or_path: str,
        language: str = "hi",
        device: str = "auto",
        chunk_length_s: int = 30,
        stride_length_s: int = 5,
        return_timestamps: bool = True,
        **kwargs,
    ):
        """
        Initialize Indic Conformer ASR model.

        Args:
            model_name_or_path: Path to the model (HuggingFace model ID)
            language: Target language code (e.g., 'hi', 'en', 'bn')
            device: Device to use for inference
            chunk_length_s: Chunk length in seconds for long audio
            stride_length_s: Stride length in seconds
            return_timestamps: Whether to return timestamps
            **kwargs: Additional model-specific parameters
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            **kwargs,
        )

        self.language = language
        self.chunk_length_s = chunk_length_s
        self.stride_length_s = stride_length_s
        self.return_timestamps = return_timestamps

    def load_model(self) -> None:
        """Load the ASR model and processor."""
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name_or_path,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )
        self.is_loaded = True

    def preprocess(
        self,
        audios: List[torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess audio inputs for the model.

        Args:
            audios: List of audio tensors
            **kwargs: Additional preprocessing arguments

        Returns:
            Preprocessed inputs ready for forward pass
        """
        # Process audio inputs
        inputs = self.processor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        return inputs.to(self.device)

    def run(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor], Any],
        **kwargs,
    ):
        """
        Forward pass through the ASR model.

        Args:
            inputs: Audio inputs (tensors or file paths)
            **kwargs: Additional forward pass arguments

        Returns:
            ASR outputs with transcriptions and optional timestamps
        """
        if not self.is_loaded:
            self.load_model()

        # Preprocess inputs
        processed_inputs = self.preprocess(inputs)

        # Generate transcriptions
        with torch.no_grad():
            predicted_ids = self.model.generate(
                **processed_inputs,
                language=self.language,
                return_timestamps=self.return_timestamps,
            )

        # Decode predictions
        transcriptions = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )

        return transcriptions

    def postprocess(self, model_outputs: Dict[str, Any], **kwargs) -> List[str]:
        """
        Postprocess model outputs into final format.

        Args:
            model_outputs: Raw model outputs from forward pass
            **kwargs: Additional postprocessing arguments

        Returns:
            Processed transcriptions
        """
        return model_outputs["transcriptions"]


# Model configurations
INDIC_CONFORMER_MULTILINGUAL_META = ModelMeta(
    name="ai4bharat/conformer_multilingual",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    description="Multilingual Conformer ASR model for Indian languages",
    n_parameters=85_000_000,
    framework=["PyTorch", "Transformers", "SpeechBrain"],
    audio_sample_rate=16000,
    supported_audio_formats=["wav", "flac", "mp3"],
    loader_class="karma.models.indic_conformer.IndicConformerASR",
    loader_kwargs={
        "language": "hi",  # Hindi by default
        "device": "auto",
        "chunk_length_s": 30,
        "stride_length_s": 5,
        "return_timestamps": True,
    },
    default_eval_kwargs={
        "language": "hi",
        "return_timestamps": True,
    },
    languages=["hin-Deva", "ben-Beng", "tam-Taml", "tel-Telu", "mar-Deva"],
    medical_domains=[
        "patient_interviews",
        "clinical_documentation",
        "telemedicine",
        "multilingual_healthcare",
    ],
    clinical_specialties=["primary_care", "psychiatry", "public_health"],
    license="MIT",
    open_weights=True,
    reference="https://huggingface.co/ai4bharat/conformer_multilingual",
    release_date="2023-06-15",
    version="1.0",
    revision=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    vision_encoder_dim=None,
    max_image_size=None,
    public_training_code=None,
    public_training_data=None,
    inference_speed_ms=None,
)

# Register model configurations
register_model_meta(INDIC_CONFORMER_MULTILINGUAL_META)
