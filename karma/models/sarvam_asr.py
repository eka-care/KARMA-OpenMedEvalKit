import logging
import os
import tempfile
from typing import List, Optional
from sarvamai import SarvamAI
from karma.models.base_model_abs import BaseModel
from karma.data_models.model_meta import ModelMeta, ModelType, ModalityType
from karma.registries.model_registry import register_model_meta
from karma.data_models.dataloader_iterable import DataLoaderIterable

logger = logging.getLogger(__name__)


class SarvamASR(BaseModel):
    """Sarvam AI ASR model for multilingual speech recognition."""

    def __init__(
        self,
        model_name_or_path: str = "saaras:v3",
        api_subscription_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Sarvam AI ASR model.

        Args:
            model_name_or_path: Model name (e.g., "saaras:v3")
            api_subscription_key: Sarvam AI API subscription key (if None, will try to get from environment)
            **kwargs: Additional model-specific parameters
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            **kwargs,
        )
        self.api_subscription_key = api_subscription_key or os.getenv("SARVAM_API_KEY")
        
        if not self.api_subscription_key:
            raise ValueError("Sarvam AI API key must be provided either as parameter or SARVAM_API_KEY environment variable")
        
        self.client = None

    def load_model(self) -> None:
        """Initialize the Sarvam AI client."""
        self.client = SarvamAI(api_subscription_key=self.api_subscription_key)
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

        if self.client is None:
            raise RuntimeError("Model is not loaded")

        transcriptions = []
        for input_item in inputs:
            transcription = self.transcribe(input_item)
            transcriptions.append(transcription)

        return transcriptions

    def transcribe(self, input_item: DataLoaderIterable) -> str:
        """
        Transcribe audio using Sarvam AI API.

        Args:
            input_item: DataLoaderIterable item containing audio bytes

        Returns:
            Transcribed text
        """
        # Create a temporary file to store the audio bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file.write(input_item.audio)
            temp_file_path = temp_file.name

        try:
            # Transcribe using Sarvam AI
            with open(temp_file_path, "rb") as audio_file:
                response = self.client.speech_to_text.transcribe(
                    file=audio_file,
                    model=self.model_name_or_path
                )
            
            # Extract text from response
            if hasattr(response, 'transcript'):
                return response.transcript
            elif isinstance(response, dict) and 'transcript' in response:
                return response['transcript']
            elif hasattr(response, 'text'):
                return response.text
            elif isinstance(response, dict) and 'text' in response:
                return response['text']
            else:
                logger.warning(f"Unexpected response format: {response}")
                return str(response)
                
        except Exception as e:
            logger.error(f"Failed to transcribe with Sarvam AI: {str(e)}")
            raise RuntimeError(f"Failed to transcribe with Sarvam AI: {str(e)}") from e
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

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
            Audio bytes (no preprocessing needed for API-based model)
        """
        return input_item.audio

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
SARVAM_ASR_V3_META = ModelMeta(
    name="saaras:v3",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    description="Sarvam AI Saaras v3 model for multilingual speech recognition",
    audio_sample_rate=16000,
    supported_audio_formats=["wav", "flac", "mp3"],
    loader_class="karma.models.sarvam_asr.SarvamASR",
    loader_kwargs={},
    default_eval_kwargs={},
    languages=["hin-Deva", "ben-Beng", "tam-Taml", "tel-Telu", "kan-Knda", "mal-Mlym", "mar-Deva", "guj-Gujr", "ori-Orya", "pan-Guru", "eng-Latn"],
    license="Proprietary",
    open_weights=False,
    reference="https://www.sarvam.ai/",
    release_date="2024-01-01",
    version="3.0",
)

SARVAM_ASR_V2_META = ModelMeta(
    name="saaras:v2",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    description="Sarvam AI Saaras v2 model for multilingual speech recognition",
    audio_sample_rate=16000,
    supported_audio_formats=["wav", "flac", "mp3"],
    loader_class="karma.models.sarvam_asr.SarvamASR",
    loader_kwargs={},
    default_eval_kwargs={},
    languages=["hin-Deva", "ben-Beng", "tam-Taml", "tel-Telu", "kan-Knda", "mal-Mlym", "mar-Deva", "guj-Gujr", "ori-Orya", "pan-Guru", "eng-Latn"],
    license="Proprietary",
    open_weights=False,
    reference="https://www.sarvam.ai/",
    release_date="2024-01-01",
    version="2.0",
)

# Register model configurations
register_model_meta(SARVAM_ASR_V3_META)
register_model_meta(SARVAM_ASR_V2_META)
