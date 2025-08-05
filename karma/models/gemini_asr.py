import tempfile
import os
from typing import List, Optional
from google.genai import types
from google import genai
from karma.models.base_model_abs import BaseModel
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.data_models.model_meta import ModelMeta, ModalityType, ModelType
from karma.registries.model_registry import register_model_meta


class GeminiASR(BaseModel):
    """Gemini-based ASR model for the KARMA framework."""
    
    def __init__(
        self, 
        model_name_or_path: str = "gemini-2.5-flash", 
        api_key: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the Gemini ASR service.
        
        Args:
            model_id: Gemini model ID to use
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
        Transcribe audio bytes using Gemini models via the Google AI API.
        
        Args:
            audio_bytes (bytes): Raw audio bytes to transcribe
            
        Returns:
            str: Transcribed text
        """
        # Create a temporary file to store the audio bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        try:
            # Upload file to Google AI
            uploaded_file = self.client.files.upload(file=temp_file_path)
            
            # Configure generation settings
            generation_config = {}
            if self.thinking_budget is not None:
                generation_config["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                )
            
            # Generate transcription
            response = self.client.models.generate_content(
                model=self.model_id,
                config=types.GenerateContentConfig(**generation_config),
                contents=[
                    'Transcribe the given audio. Instruction: 1. Do not generate any timestamps or speaker information, just provide the text.',
                    uploaded_file
                ]
            )
            
            return response.text if response.text else ""
            
        except Exception as e:
            raise RuntimeError(f"Failed to transcribe with Gemini: {str(e)}") from e
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
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
GeminiASR_2_0_Flash = ModelMeta(
    name="gemini-2.0-flash",
    description="Google Gemini 2.0 Flash ASR model",
    loader_class="karma.models.gemini_asr.GeminiASR",
    loader_kwargs={
        "model_id": "gemini-2.0-flash",
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
        "model_id": "gemini-2.5-flash",
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
        "model_id": "gemini-2.5-flash-lite",
        "thinking_budget": None,
    },
    revision=None,
    reference="https://ai.google.dev/gemini-api/docs",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    release_date="2024-05-14",
    version="2.5",
)

# Register the models
register_model_meta(GeminiASR_2_0_Flash)
register_model_meta(GeminiASR_2_5_Flash)
register_model_meta(GeminiASR_2_5_Flash_lite)