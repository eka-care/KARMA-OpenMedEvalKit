import logging
import os
from typing import List
import torch
import librosa
from transformers import AutoModel
from io import BytesIO
from karma.models.base_model_abs import BaseModel
from karma.data_models.model_meta import ModelMeta, ModelType, ModalityType
from karma.registries.model_registry import register_model_meta
from karma.data_models.dataloader_iterable import DataLoaderIterable

logger = logging.getLogger(__name__)


class IndicConformerASR(BaseModel):
    """Indic Conformer ASR model for multilingual speech recognition."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        language: str = "hi",
        decoding_method: str = "ctc",
        target_sample_rate: int = 16000,
        **kwargs,
    ):
        """
        Initialize Indic Conformer ASR model.

        Args:
            model_name_or_path: Path to the model (HuggingFace model ID)
            device: Device to use for inference
            language: Target language code (e.g., 'hi', 'en', 'bn')
            decoding_method: Decoding method ('ctc' or 'rnnt')
            target_sample_rate: Expected sample rate for audio
            **kwargs: Additional model-specific parameters
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            **kwargs,
        )

        self.language = language
        self.decoding_method = decoding_method
        self.target_sample_rate = target_sample_rate

    def load_model(self) -> None:
        """Load the ASR model."""
        # login to HF
        from huggingface_hub import login

        try:
            login(os.getenv("HUGGINGFACE_TOKEN"))
        except ValueError:
            logger.warning("HF token not found, will not login.")

        self.model = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )

        # Move model to device using base class method
        self.to(self.device)
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
            language = (
                input_item.other_args["language"][:2].lower()
                if input_item.other_args is not None
                else self.language
            )
            decoding_method = kwargs.get("decoding_method", self.decoding_method)

            processed_audio = self.preprocess(input_item)

            processed_audio = processed_audio.to(self.device)

            with torch.no_grad():
                transcription = self.model(processed_audio, language, decoding_method)

            transcriptions.append(transcription)

        return transcriptions

    def preprocess(
        self,
        input_item: DataLoaderIterable,
        **kwargs,
    ) -> torch.Tensor:
        """
        Preprocess inputs for compatibility with base class interface.

        Args:
            inputs: List of DataLoaderIterable items
            **kwargs: Additional preprocessing arguments

        Returns:
            List of preprocessed audio tensors
        """
        audio, _ = librosa.load(BytesIO(input_item.audio), sr=self.target_sample_rate)
        wav_tensor = (
            torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        return wav_tensor

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
INDIC_CONFORMER_MULTILINGUAL_META = ModelMeta(
    name="ai4bharat/indic-conformer-600m-multilingual",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    description="Multilingual Conformer ASR model for Indian languages",
    loader_class="karma.models.indic_conformer.IndicConformerASR",
    loader_kwargs={
        "language": "hi",  # Hindi by default
        "device": "cpu",
        "decoding_method": "ctc",
        "target_sample_rate": 16000,
    },
    default_eval_kwargs={
        "language": "hi",
        "decoding_method": "ctc",
    },
    languages=["hin-Deva", "ben-Beng", "tam-Taml", "tel-Telu", "mar-Deva"],
    reference="https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual",
    release_date="2023-06-15",
    version="1.0",
)

# Register model configurations
register_model_meta(INDIC_CONFORMER_MULTILINGUAL_META)
