import logging
from typing import List
from io import BytesIO
import librosa
import torch
from karma.models.base_model_abs import BaseModel
from karma.data_models.model_meta import ModelMeta, ModelType, ModalityType
from karma.registries.model_registry import register_model_meta
from karma.data_models.dataloader_iterable import DataLoaderIterable
import os
logger = logging.getLogger(__name__)


class MedASR(BaseModel):
    """Google MedASR model for medical speech recognition using Hugging Face transformers."""

    def __init__(
        self,
        model_name_or_path: str = "google/medasr",
        **kwargs,
    ):
        """
        Initialize MedASR model.

        Args:
            model_name_or_path: Path to the model (HuggingFace model ID)
            **kwargs: Additional model-specific parameters
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            **kwargs,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.install_dependeicies()
        

    def install_dependeicies(self) -> None:
        os.system("uv pip uninstall transformers")
        os.system("uv pip install git+https://github.com/huggingface/transformers.git@65dc261512cbdb1ee72b88ae5b222f2605aad8e5")
        

    def load_model(self) -> None:
        """Load the ASR model using AutoModelForCTC and AutoProcessor."""
        from transformers import AutoModelForCTC, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForCTC.from_pretrained(self.model_name_or_path).to(self.device)
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

        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded")

        transcriptions = []
        for input_item in inputs:
            speech, sample_rate = self.preprocess(input_item)
            
            # Process audio
            processor_inputs = self.processor(
                speech, 
                sampling_rate=sample_rate, 
                return_tensors="pt", 
                padding=True
            )
            processor_inputs = processor_inputs.to(self.device)
            
            # Generate transcription
            outputs = self.model.generate(**processor_inputs)
            decoded_text = self.processor.batch_decode(outputs)[0]
            transcriptions.append(decoded_text.replace("<epsilon>", "").replace("</s>", ""))

        return transcriptions

    def preprocess(
        self,
        input_item: DataLoaderIterable,
        **kwargs,
    ):
        """
        Preprocess inputs for compatibility with base class interface.

        Args:
            input_item: DataLoaderIterable item containing audio bytes
            **kwargs: Additional preprocessing arguments

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Load audio from bytes using librosa
        speech, sample_rate = librosa.load(BytesIO(input_item.audio), sr=16000)
        return speech, sample_rate

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
MEDASR_META = ModelMeta(
    name="google/medasr",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    description="Google MedASR model for medical speech recognition",
    audio_sample_rate=16000,
    supported_audio_formats=["wav", "flac", "mp3"],
    loader_class="karma.models.medasr.MedASR",
    loader_kwargs={},
    default_eval_kwargs={},
    languages=["eng-Latn"],
    license="Apache-2.0",
    open_weights=True,
    reference="https://huggingface.co/google/medasr",
    release_date="2024-01-01",
    version="1.0",
)

# Register model configurations
register_model_meta(MEDASR_META)
