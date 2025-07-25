from karma.models.base_model_abs import BaseModel
from elevenlabs import ElevenLabs
from typing import List
from karma.data_models.dataloader_iterable import DataLoaderIterable
import os
from karma.data_models.model_meta import ModelMeta, ModalityType, ModelType
from karma.registries.model_registry import register_model_meta

class ElevenLabsASR(BaseModel):
    def __init__(self, model_name_or_path: str = "scribe_v1"):
        super().__init__(
            model_name_or_path=model_name_or_path,
        )

        self.config = {
                "diarize": False,
                "tag_audio_events": False,
                "model_id": model_name_or_path
            }
        self.load_model()
        
    def load_model(self):
        self.client = ElevenLabs(api_key=os.getenv("ELEVEN_LABS_API_KEY"))

    def preprocess(self, inputs: List[DataLoaderIterable], **kwargs):
        audio_items = []
        for item in inputs:
            audio_items.append(item.audio)
        return audio_items

    def run(self, inputs: List[DataLoaderIterable], **kwargs):
        transcriptions = []
        audio_items = self.preprocess(inputs)
        for audio_item in audio_items:
            transcriptions.append(self.transcribe(audio_item))
        return transcriptions


    def transcribe(self, audio_item):
        transcription = self.client.speech_to_text.convert(
            file=audio_item,
            **self.config
        )
        return transcription.text

    def postprocess(self, transcriptions: List[str], **kwargs):
        return transcriptions

ElevenLabsASRModel = ModelMeta(
    name="scribe_v1",
    description="ElevenLabs ASR model",
    loader_class="karma.models.eleven_labs.ElevenLabsASR",
    revision=None,
    reference=None,
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO]
)
register_model_meta(ElevenLabsASRModel)