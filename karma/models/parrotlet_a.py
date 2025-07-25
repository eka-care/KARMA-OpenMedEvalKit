import logging
import os
from typing import List, Tuple
import torch
import librosa
from transformers import AutoModel
from io import BytesIO
from karma.models.base_model_abs import BaseModel
from karma.data_models.model_meta import ModelMeta, ModelType, ModalityType
from karma.registries.model_registry import register_model_meta
from karma.data_models.dataloader_iterable import DataLoaderIterable

logger = logging.getLogger(__name__)


class ParrotletASR(BaseModel):
    """Parrotlet ASR model for multilingual speech recognition."""

    def __init__(
        self,
        model_name_or_path: str,
        **kwargs,
    ):
        """
        Initialize Parrotlet ASR model.

        Args:
            model_name_or_path: Path to the model (HuggingFace model ID)
            **kwargs: Additional model-specific parameters
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            **kwargs,
        )

    def load_model(self) -> None:
        """Load the ASR model."""
        self.model = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )

        self.is_loaded = True

    def run(
        self,
        inputs: List[DataLoaderIterable],
        **kwargs,
    ) -> List[str]:
        """
        Forward pass through the ASR model.

        Args:
            inputs: Audio inputs with DataLoaderIterable format
            **kwargs: Additional forward pass arguments

        Returns:
            List of transcriptions
        """
        if not self.is_loaded:
            self.load_model()

        if self.model is None:
            raise RuntimeError("Model is not loaded")

        transcriptions = []

        for input_item in inputs:

            processed_audio, sr = self.preprocess(input_item)

            transcription = self.model.transcribe(processed_audio, sr)

            transcriptions.append(transcription)

        return transcriptions

    def preprocess(
        self,
        input_item: DataLoaderIterable,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """
        Preprocess inputs for compatibility with base class interface.

        Args:
            inputs: List of DataLoaderIterable items
            **kwargs: Additional preprocessing arguments

        Returns:
            List of preprocessed audio tensors
        """
        audio, sr = librosa.load(BytesIO(input_item.audio), sr=16000)
        return audio, sr

    def postprocess(self, model_outputs: List[str], **kwargs) -> List[str]:
        """
        Postprocess model outputs into final format.

        Args:
            model_outputs: Raw model outputs from forward pass
            **kwargs: Additional postprocessing arguments

        Returns:
            Processed transcriptions (already strings in this case)
        """
        return model_outputs


# Model configurations
PARROTL_A_META = ModelMeta(
    name="ekacare/parrotlet-a-en-5b",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    description="Parrotlet ASR model for English",
    audio_sample_rate=16000,
    supported_audio_formats=["wav", "flac", "mp3"],
    loader_class="karma.models.parrotlet_a.ParrotletASR",
    loader_kwargs={},
    default_eval_kwargs={},
    languages=["eng-Latn"],
    license="MIT",
    open_weights=True,
    reference="https://huggingface.co/ekacare/parrotlet-a-en-5b",
    release_date="2025-07-25",
    version="1.0",
)

# Register model configurations
register_model_meta(PARROTL_A_META)
