import os
import io
import logging
from typing import List, Optional
from openai import OpenAI
from karma.models.base_model_abs import BaseModel
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.data_models.model_meta import ModelMeta, ModalityType, ModelType
from karma.registries.model_registry import register_model_meta
from karma.utils.retry import call_with_retry

logger = logging.getLogger(__name__)


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

    def detect_audio_format(self, audio_bytes: bytes) -> str:
        """
        Detect audio format from bytes using magic numbers (file signatures).
        
        Args:
            audio_bytes: Raw audio bytes
            
        Returns:
            str: Detected file extension (e.g., 'mp3', 'wav', 'flac', etc.)
        """
        if not audio_bytes or len(audio_bytes) < 12:
            return "mp3"  # Default fallback
        
        # Check for various audio format signatures
        if audio_bytes.startswith(b'RIFF') and b'WAVE' in audio_bytes[8:12]:
            return "wav"
        elif audio_bytes.startswith(b'ID3') or audio_bytes.startswith(b'\xff\xfb'):
            return "mp3"
        elif audio_bytes.startswith(b'\xff\xf1') or audio_bytes.startswith(b'\xff\xf9'):
            return "aac"
        elif audio_bytes.startswith(b'fLaC'):
            return "flac"
        elif audio_bytes.startswith(b'OggS'):
            return "ogg"
        elif audio_bytes[4:8] == b'ftyp':
            # M4A/MP4 container format
            return "m4a"
        elif audio_bytes.startswith(b'FORM') and b'AIFF' in audio_bytes[8:12]:
            return "aiff"
        elif audio_bytes.startswith(b'.snd'):
            return "au"
        elif audio_bytes.startswith(b'wvpk'):
            return "wv"  # WavPack
        else:
            # Try to detect MP3 by looking for frame sync
            for i in range(min(1024, len(audio_bytes) - 1)):
                if audio_bytes[i] == 0xff and (audio_bytes[i + 1] & 0xe0) == 0xe0:
                    return "mp3"
            
            # Default fallback
            return "wav"
    
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
            List of {"text", "error"} dicts, one per input
        """
        results = []
        for i, item in enumerate(inputs):
            try:
                results.append({"text": self.transcribe(item.audio, item.input), "error": None})
            except Exception as e:
                logger.warning("OpenAI transcription failed for input %d: %s", i, e)
                results.append({"text": "", "error": str(e)})
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

    def transcribe(self, audio_bytes, instruction=None):
        """
        Transcribe audio bytes using OpenAI models via the OpenAI API.

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
                "OpenAIASR received audio with no instruction. Set a user_prompt "
                "on the dataset (e.g. default_args={'user_prompt': 'asr_transcribe'}) "
                "or pass --dataset-args '<dataset>:user_prompt_text=...'."
            )
        try:
            detected_format = self.detect_audio_format(audio_bytes)

            def _call():
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = f"audio.{detected_format}"
                transcription = self.client.audio.transcriptions.create(
                    model=self.model_id,
                    file=audio_file,
                    prompt=instruction
                )
                return transcription.text if transcription.text else ""

            return call_with_retry(_call, description="OpenAI transcribe")
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
    },
    revision=None,
    reference="https://platform.openai.com/docs/guides/speech-to-text",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    release_date="2025-06-20",
    version="1.0",
)

GPTTranscribe_ASR = ModelMeta(
    name="gpt-transcribe",
    description="OpenAI GPT-Transcribe speech-to-text model (successor to gpt-4o-transcribe)",
    loader_class="karma.models.openai_asr.OpenAIASR",
    loader_kwargs={
    },
    revision=None,
    reference="https://developers.openai.com/api/docs/models/gpt-transcribe",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    release_date="2026-07-28",
    version="1.0",
)

# Register the model
register_model_meta(GPT4o_ASR)
register_model_meta(GPTTranscribe_ASR)