import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from karma.models.base_model_abs import BaseModel
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.data_models.model_meta import ModelMeta, ModalityType, ModelType
from karma.registries.model_registry import register_model_meta
from karma.utils.retry import call_with_retry

logger = logging.getLogger(__name__)


class DeepgramASR(BaseModel):
    """Deepgram-based ASR model for the KARMA framework.

    Sends audio bytes to Deepgram's REST API and returns transcripts.
    Supports all Deepgram pre-recorded audio models (nova-3, nova-3-medical, etc.).
    Auth via DEEPGRAM_API_KEY env var or api_key constructor param.
    """

    def __init__(
        self,
        model_name_or_path: str = "nova-3",
        api_key: Optional[str] = None,
        language: Optional[str] = None,
        smart_format: bool = True,
        **kwargs,
    ):
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)
        self.model_id = model_name_or_path
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Deepgram API key must be provided as api_key param or DEEPGRAM_API_KEY env var"
            )
        self.language = language
        self.smart_format = smart_format
        self.client = None
        self.load_model()

    def load_model(self) -> None:
        from deepgram import DeepgramClient
        self.client = DeepgramClient(api_key=self.api_key)
        self.is_loaded = True

    def transcribe(self, audio_bytes: bytes) -> str:
        transcribe_kwargs = {
            "request": audio_bytes,
            "model": self.model_id,
            "smart_format": self.smart_format,
        }
        if self.language:
            transcribe_kwargs["language"] = self.language

        try:
            response = call_with_retry(
                lambda: self.client.listen.v1.media.transcribe_file(**transcribe_kwargs),
                description="Deepgram transcribe",
            )
            return response.results.channels[0].alternatives[0].transcript or ""
        except Exception as e:
            raise RuntimeError(f"Deepgram transcription failed: {e}") from e

    def run(self, inputs: List[DataLoaderIterable], max_workers: int = 10, **kwargs):
        if not self.is_loaded:
            self.load_model()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.transcribe, item.audio): i
                for i, item in enumerate(inputs)
            }
            results = [None] * len(inputs)
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = {"text": future.result(), "error": None}
                except Exception as e:
                    logger.warning("Deepgram transcription failed for input %d: %s", idx, e)
                    results[idx] = {"text": "", "error": str(e)}

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

    def preprocess(self, inputs: List[DataLoaderIterable], **kwargs):
        return inputs

    def postprocess(self, outputs: List[str], **kwargs):
        return outputs


DeepgramNova3 = ModelMeta(
    name="deepgram-nova-3",
    description="Deepgram Nova-3 general-purpose ASR model",
    loader_class="karma.models.deepgram_asr.DeepgramASR",
    model_path="nova-3",
    loader_kwargs={"smart_format": True},
    revision=None,
    reference="https://developers.deepgram.com/docs/models-languages-overview",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    release_date="2024-11-01",
    version="3",
)

DeepgramNova3Medical = ModelMeta(
    name="deepgram-nova-3-medical",
    description="Deepgram Nova-3 Medical ASR model, tuned for clinical and medical terminology",
    loader_class="karma.models.deepgram_asr.DeepgramASR",
    model_path="nova-3-medical",
    loader_kwargs={"smart_format": True},
    revision=None,
    reference="https://developers.deepgram.com/docs/models-languages-overview",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    release_date="2024-11-01",
    version="3",
)

register_model_meta(DeepgramNova3)
register_model_meta(DeepgramNova3Medical)
