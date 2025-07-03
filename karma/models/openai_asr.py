import os
import io
from typing import List, Optional
from openai import OpenAI
from karma.models.base_model_abs import BaseModel
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.data_models.model_meta import ModelMeta, ModalityType, ModelType
from karma.registries.model_registry import register_model_meta


class OpenAIASR(BaseModel):
    """OpenAI-based ASR model for the KARMA framework."""
    
    def __init__(
        self, 
        model_name_or_path: str = "gpt-4o-transcribe", 
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the OpenAI ASR service.
        
        Args:
            model_id: OpenAI model ID to use (e.g., "gpt-4o-transcribe")
            api_key: OpenAI API key (if None, will try to get from environment)
            **kwargs: Additional arguments passed to BaseModel
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            **kwargs,
        )
        
        self.model_id = model_name_or_path
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either as parameter or OPENAI_API_KEY environment variable")
        
        self.client = None
        self.load_model()
    
    def load_model(self):
        """Initialize the OpenAI client."""
        self.client = OpenAI(api_key=self.api_key)
        self.is_loaded = True
    
    def preprocess(self, inputs: List[DataLoaderIterable], **kwargs):
        """
        Preprocess audio inputs for transcription.
        
        Args:
            inputs: List of DataLoaderIterable objects containing audio data
            
        Returns:
            List of audio items ready for processing
        """
        audio_items = []
        for item in inputs:
            audio_items.append(item.audio)
        return audio_items
    
    def run(self, inputs: List[DataLoaderIterable], **kwargs):
        """
        Run transcription on the input audio files.
        
        Args:
            inputs: List of DataLoaderIterable objects containing audio data
            
        Returns:
            List of transcribed text strings
        """
        transcriptions = []
        audio_items = self.preprocess(inputs)
        
        for audio_item in audio_items:
            transcription = self.transcribe(audio_item)
            transcriptions.append(transcription)
        
        return transcriptions
    
    def transcribe(self, audio_bytes):
        """
        Transcribe audio bytes using OpenAI models via the OpenAI API.
        
        Args:
            audio_bytes (bytes): Raw audio bytes to transcribe
            
        Returns:
            str: Transcribed text
        """
        try:
            # Create an in-memory file-like object from bytes
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"  # OpenAI API needs a filename for format detection
            
            # Create transcription using OpenAI Whisper
            transcription = self.client.audio.transcriptions.create(
                model=self.model_id,
                file=audio_file,
                prompt="Transcribe the given audio. Provide only the text without timestamps or speaker information."
            )
            
            return transcription.text if transcription.text else ""
            
        except Exception as e:
            raise RuntimeError(f"Failed to transcribe with OpenAI: {str(e)}") from e
    
    def postprocess(self, transcriptions: List[str], **kwargs):
        """
        Postprocess transcriptions (currently just returns them as-is).
        
        Args:
            transcriptions: List of transcribed text strings
            
        Returns:
            List of processed transcriptions
        """
        return transcriptions


# Model metadata definitions
GPT4o_ASR = ModelMeta(
    name="gpt-4o-transcribe",
    description="OpenAI GPT-4o ASR model",
    loader_class="karma.models.openai_asr.OpenAIASR",
    loader_kwargs={
        "model_id": "gpt-4o-transcribe",
    },
    revision=None,
    reference="https://platform.openai.com/docs/guides/speech-to-text",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    framework=["openai"],
    audio_sample_rate=16000,
    supported_audio_formats=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
    vision_encoder_dim=None,
    max_image_size=None,
    inference_speed_ms=None,
    release_date="2025-06-20",
    version="1.0",
    license=None,
    open_weights=False,
    public_training_code=None,
    public_training_data=None,
)

# Register the model
register_model_meta(GPT4o_ASR) 