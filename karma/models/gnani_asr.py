import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import requests

from karma.models.base_model_abs import BaseModel
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.data_models.model_meta import ModelMeta, ModalityType, ModelType
from karma.registries.model_registry import register_model_meta
from karma.utils.retry import call_with_retry, is_transient_error

logger = logging.getLogger(__name__)

GNANI_STT_URL = "https://api.vachana.ai/stt/v3"

# Normalized ISO 639-1 sample language -> Gnani BCP-47 language_code
LANGUAGE_CODE_MAP = {
    "bn": "bn-IN",
    "en": "en-IN",
    "gu": "gu-IN",
    "hi": "hi-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "pa": "pa-IN",
    "ta": "ta-IN",
    "te": "te-IN",
}


def _sniff_audio_extension(audio: bytes) -> str:
    """The Vachana API infers the audio format from the uploaded filename's
    extension, so it must match the actual byte content."""
    if audio[:4] == b"RIFF":
        return "wav"
    if audio[:4] == b"fLaC":
        return "flac"
    if audio[:4] == b"OggS":
        return "ogg"
    if audio[4:8] == b"ftyp":
        return "m4a"
    return "mp3"


class GnaniASR(BaseModel):
    """Gnani.ai Prisma ASR model for the KARMA framework.

    Sends audio bytes to Gnani's Vachana STT REST API (multipart upload,
    clips up to 60 seconds) and returns native-script transcripts.
    Auth via GNANI_API_KEY env var or api_key constructor param.
    The per-sample language code is taken from DataLoaderIterable.language
    (falling back to the `language` constructor param).
    """

    def __init__(
        self,
        model_name_or_path: str = "prisma-v2.5",
        api_key: Optional[str] = None,
        language: Optional[str] = None,
        output_format: str = "verbatim",
        itn_native_numerals: bool = False,
        **kwargs,
    ):
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)
        self.api_key = api_key or os.getenv("GNANI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gnani API key must be provided as api_key param or GNANI_API_KEY env var"
            )
        self.language = language
        self.output_format = output_format
        self.itn_native_numerals = itn_native_numerals
        self.session = None
        self.load_model()

    def load_model(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key-ID": self.api_key})
        self.is_loaded = True

    def _resolve_language_code(self, item: DataLoaderIterable) -> str:
        language = item.language or self.language
        if not language:
            raise ValueError(
                "Gnani STT requires a language; sample has no language and no default was set"
            )
        code = LANGUAGE_CODE_MAP.get(language, language)
        if code not in LANGUAGE_CODE_MAP.values():
            raise ValueError(f"Language {language!r} is not supported by Gnani STT")
        return code

    def transcribe(self, item: DataLoaderIterable) -> str:
        language_code = self._resolve_language_code(item)
        extension = _sniff_audio_extension(item.audio)
        data = {
            "language_code": language_code,
            "format": self.output_format,
            "itn_native_numerals": str(self.itn_native_numerals).lower(),
        }

        def _call():
            response = self.session.post(
                GNANI_STT_URL,
                data=data,
                files={"audio_file": (f"audio.{extension}", item.audio)},
                timeout=120,
            )
            response.raise_for_status()
            return response.json().get("transcript") or ""

        return call_with_retry(
            _call,
            is_retryable=lambda e: isinstance(e, requests.ConnectionError)
            or is_transient_error(e),
            description="Gnani transcribe",
        )

    def run(self, inputs: List[DataLoaderIterable], max_workers: int = 10, **kwargs):
        if not self.is_loaded:
            self.load_model()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.transcribe, item): i
                for i, item in enumerate(inputs)
            }
            results = [None] * len(inputs)
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = {"text": future.result(), "error": None}
                except Exception as e:
                    logger.warning(
                        "Gnani transcription failed for input %d: %s", idx, e
                    )
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


GnaniPrismaV25 = ModelMeta(
    name="gnani-prisma-v2.5",
    description="Gnani.ai Prisma v2.5 multilingual Indic ASR model (Vachana STT REST API)",
    loader_class="karma.models.gnani_asr.GnaniASR",
    model_path="prisma-v2.5",
    loader_kwargs={"output_format": "verbatim"},
    revision=None,
    reference="https://docs.gnani.ai/api/STT/speech-to-text",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    supported_audio_formats=["wav", "mp3", "ogg", "flac", "aac", "m4a"],
    languages=[
        "ben-Beng",
        "eng-Latn",
        "guj-Gujr",
        "hin-Deva",
        "kan-Knda",
        "mal-Mlym",
        "mar-Deva",
        "pan-Guru",
        "tam-Taml",
        "tel-Telu",
    ],
    license="Proprietary",
    open_weights=False,
    release_date="2026-06-19",
    version="2.5",
)

register_model_meta(GnaniPrismaV25)
