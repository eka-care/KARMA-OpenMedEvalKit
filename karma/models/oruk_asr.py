"""Oruk Fourier 2B ASR adapters for the KARMA evaluation framework.

Provides two adapters:
1. ``OrukASR`` — Hosted OpenAI-compatible REST API (`https://speech-api.oruk.ai/v1`,
   ``model="oruk-fourier-2b"``, auth via ``ORUK_API_KEY``). Uses the existing
   ``openai`` dependency with zero extra installs.
2. ``OrukFourierLocalASR`` — Local open-weight native Transformers BF16 inference
   for ``oruk/oruk-fourier-2b`` (pinned revision
   ``8c2a8871f1304aa2f634da4fb95e24438c4945b9``, ``CC BY-NC 4.0`` non-commercial
   weights; requires ``transformers>=5.24.0``).
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlparse

import numpy as np

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.data_models.model_meta import ModalityType, ModelMeta, ModelType
from karma.models.base_model_abs import BaseModel
from karma.registries.model_registry import register_model_meta
from karma.utils.retry import call_with_retry

logger = logging.getLogger(__name__)

ORUK_HOSTED_BASE_URL = "https://speech-api.oruk.ai/v1"
ORUK_FOURIER_MODEL_ID = "oruk/oruk-fourier-2b"
ORUK_FOURIER_REVISION = "8c2a8871f1304aa2f634da4fb95e24438c4945b9"
TARGET_SAMPLE_RATE = 16_000
DEFAULT_CHUNK_SECONDS = 30.0


def _validate_base_url(base_url: str) -> str:
    """Ensure hosted base_url uses HTTPS (or localhost/loopback HTTP for tests)."""
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "https" and host:
        return base_url.rstrip("/")
    if parsed.scheme == "http" and host in {"localhost", "127.0.0.1", "::1"}:
        return base_url.rstrip("/")
    raise ValueError(
        f"Invalid Oruk base_url {base_url!r}: must use https:// (or loopback http:// for local testing)"
    )


def _detect_audio_format(audio_bytes: bytes) -> str:
    """Detect audio container extension from magic bytes."""
    if not audio_bytes or len(audio_bytes) < 12:
        return "wav"
    if audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[8:12]:
        return "wav"
    if audio_bytes.startswith((b"ID3", b"\xff\xfb")):
        return "mp3"
    if audio_bytes.startswith(b"fLaC"):
        return "flac"
    if audio_bytes.startswith(b"OggS"):
        return "ogg"
    if audio_bytes[4:8] == b"ftyp":
        return "m4a"
    if audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):
        return "webm"
    return "wav"


def _redact_secret(message: str, secret: str | None) -> str:
    if secret and secret in message:
        return message.replace(secret, "[REDACTED]")
    return message


def _decode_audio_bytes_16k(audio_bytes: bytes) -> np.ndarray:
    """Decode raw audio bytes to 16 kHz mono float32 without TorchCodec."""
    samples: np.ndarray | None = None
    sr: int = TARGET_SAMPLE_RATE

    try:
        import soundfile as sf

        samples, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    except Exception as decode_err:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                str(TARGET_SAMPLE_RATE),
                "pipe:1",
            ],
            input=audio_bytes,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Failed to decode audio bytes via soundfile and ffmpeg: {err}"
            ) from decode_err
        samples = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        sr = TARGET_SAMPLE_RATE

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = (
            samples.mean(axis=0)
            if samples.shape[0] <= 8 and samples.shape[1] > samples.shape[0]
            else samples.mean(axis=1)
        )
    if samples.size == 0 or sr <= 0:
        raise ValueError(f"Decoded empty audio payload (shape={samples.shape}, sr={sr})")

    if sr != TARGET_SAMPLE_RATE:
        try:
            import soxr

            samples = soxr.resample(samples, sr, TARGET_SAMPLE_RATE).astype(np.float32, copy=False)
        except ImportError:
            import librosa

            samples = librosa.resample(samples, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE).astype(
                np.float32, copy=False
            )
    return samples


def _split_chunks(waveform: np.ndarray, chunk_seconds: float = DEFAULT_CHUNK_SECONDS) -> list[np.ndarray]:
    """Split long 16 kHz audio into non-overlapping chunks."""
    max_samples = int(chunk_seconds * TARGET_SAMPLE_RATE)
    if waveform.size <= max_samples:
        return [waveform]
    return [
        waveform[i : i + max_samples]
        for i in range(0, waveform.size, max_samples)
        if waveform[i : i + max_samples].size > 0
    ]


ORUK_SUPPORTED_API_LANGUAGES = frozenset({"en", "en-US", "en-GB", "english", "auto"})


class OrukASR(BaseModel):
    """Hosted Oruk Speech API adapter (OpenAI-compatible ``/v1/audio/transcriptions``).

    Targets ``https://speech-api.oruk.ai/v1`` with ``model="oruk-fourier-2b"`` by default.
    Auth via ``ORUK_API_KEY`` environment variable or ``api_key`` parameter.
    Follows ``Fourier2bRequest`` (``file``, ``model``, ``language``, ``response_format``).
    """

    def __init__(
        self,
        model_name_or_path: str = "oruk-fourier-2b",
        api_key: str | None = None,
        base_url: str | None = None,
        language: str = "en",
        timeout: float = 120.0,
        **kwargs,
    ):
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)
        if language not in ORUK_SUPPORTED_API_LANGUAGES:
            raise ValueError(
                f"Unsupported Oruk Speech API language {language!r}; "
                f"expected one of {sorted(ORUK_SUPPORTED_API_LANGUAGES)}"
            )
        self.model_id = model_name_or_path
        self.api_key = api_key or os.getenv("ORUK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Oruk API key must be provided as api_key parameter or ORUK_API_KEY environment variable"
            )
        self.base_url = _validate_base_url(
            base_url or os.getenv("ORUK_BASE_URL", ORUK_HOSTED_BASE_URL)
        )
        self.language = language
        self.timeout = float(timeout)
        self.client = None
        self.load_model()

    def load_model(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        self.is_loaded = True

    def preprocess(self, inputs: list[DataLoaderIterable], **kwargs) -> list[bytes]:
        return [item.audio for item in inputs]

    def postprocess(self, transcriptions: list[str], **kwargs) -> list[str]:
        return transcriptions

    def transcribe(self, audio_bytes: bytes) -> str:
        if not self.is_loaded or self.client is None:
            self.load_model()

        ext = _detect_audio_format(audio_bytes)

        def _call() -> str:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = f"audio.{ext}"
            req_kwargs: dict[str, Any] = {
                "model": self.model_id,
                "file": audio_file,
                "language": self.language,
            }
            resp = self.client.audio.transcriptions.create(**req_kwargs)
            return (getattr(resp, "text", None) or "").strip()

        try:
            return call_with_retry(_call, description="Oruk ASR transcribe")
        except Exception as exc:
            safe_msg = _redact_secret(str(exc), self.api_key)
            raise RuntimeError(f"Oruk ASR transcription failed: {safe_msg}") from exc

    def run(
        self,
        inputs: list[DataLoaderIterable],
        max_workers: int = 8,
        **kwargs,
    ) -> list[dict[str, str | None]]:
        if not self.is_loaded:
            self.load_model()

        results: list[dict[str, str | None] | None] = [None] * len(inputs)
        worker_count = max(1, min(int(max_workers), len(inputs) or 1))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(self.transcribe, item.audio): idx
                for idx, item in enumerate(inputs)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = {"text": fut.result(), "error": None}
                except Exception as exc:  # noqa: BLE001
                    safe_err = _redact_secret(str(exc), self.api_key)
                    logger.warning("Oruk ASR transcription failed for input %d: %s", idx, safe_err)
                    results[idx] = {"text": "", "error": safe_err}
        return [r if r is not None else {"text": "", "error": "empty"} for r in results]

    def run_with_metadata(self, inputs: list[DataLoaderIterable], **kwargs):
        outputs = self.run(inputs, **kwargs)
        return [
            {
                "text": (out.get("text") or "").strip(),
                "tool_trace": "",
                "error": out.get("error"),
            }
            for out in outputs
        ]


class OrukFourierLocalASR(BaseModel):
    """Local open-weight native Transformers BF16 adapter for ``oruk/oruk-fourier-2b``.

    Downloads and loads directly from the canonical Hugging Face repo
    ``oruk/oruk-fourier-2b`` (pinned default revision ``8c2a8871...``) via
    ``AutoProcessor`` + ``AutoModelForMultimodalLM`` (requires ``transformers>=5.24.0``).
    Weights are licensed under ``CC BY-NC 4.0`` (non-commercial research and personal use).
    """

    def __init__(
        self,
        model_name_or_path: str = ORUK_FOURIER_MODEL_ID,
        revision: str | None = ORUK_FOURIER_REVISION,
        decode_mode: str = "published",
        language: str = "English",
        prompt: str | None = None,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        max_new_tokens: int | None = None,
        **kwargs,
    ):
        if decode_mode not in {"published", "greedy"}:
            raise ValueError(f"Unsupported decode_mode={decode_mode!r}; use 'published' or 'greedy'")
        super().__init__(model_name_or_path=model_name_or_path, revision=revision, **kwargs)
        self.decode_mode = decode_mode
        self.language = language
        self.prompt = prompt
        self.chunk_seconds = float(chunk_seconds)
        self.max_new_tokens = max_new_tokens
        self.processor = None
        self.model = None

    def _generation_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.decode_mode == "greedy":
            kwargs["do_sample"] = False
            kwargs["num_beams"] = 1
        if self.max_new_tokens is not None:
            kwargs["max_new_tokens"] = int(self.max_new_tokens)
        return kwargs

    def load_model(self) -> None:
        import torch

        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "OrukFourierLocalASR requires `transformers>=5.24.0` with native Qwen3-ASR "
                "support (`pip install 'transformers>=5.24.0'`). For stock KARMA environments, "
                "use the hosted `oruk-fourier-2b` API model instead."
            ) from exc

        dtype = torch.bfloat16 if str(self.device).startswith(("cuda", "mps")) else torch.float32
        hub_kwargs: dict[str, Any] = {}
        if self.revision:
            hub_kwargs["revision"] = self.revision

        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path, **hub_kwargs)
        self.processor.tokenizer.padding_side = "left"
        self.model = (
            AutoModelForMultimodalLM.from_pretrained(
                self.model_name_or_path,
                dtype=dtype,
                **hub_kwargs,
            )
            .to(self.device)
            .eval()
        )
        self.is_loaded = True

    def preprocess(self, input_item: DataLoaderIterable, **kwargs) -> tuple[np.ndarray, int]:
        return _decode_audio_bytes_16k(input_item.audio), TARGET_SAMPLE_RATE

    def postprocess(self, model_outputs: list[str], **kwargs) -> list[str]:
        return [str(t).strip() for t in model_outputs]

    def run(self, inputs: list[DataLoaderIterable], **kwargs) -> list[str]:
        import torch

        if not self.is_loaded:
            self.load_model()
        if self.model is None or self.processor is None:
            raise RuntimeError("OrukFourierLocalASR model is not loaded")

        gen_kwargs = self._generation_kwargs()
        transcriptions: list[str] = []

        with torch.inference_mode():
            for item in inputs:
                waveform, _ = self.preprocess(item)
                chunks = _split_chunks(waveform, chunk_seconds=self.chunk_seconds)
                pieces: list[str] = []
                for chunk in chunks:
                    req_kwargs: dict[str, Any] = {
                        "audio": chunk,
                        "language": self.language,
                        "return_tensors": "pt",
                    }
                    if self.prompt:
                        req_kwargs["prompt"] = self.prompt
                    proc_inputs = self.processor.apply_transcription_request(**req_kwargs)
                    proc_inputs = proc_inputs.to(self.model.device, self.model.dtype)
                    out_ids = self.model.generate(**proc_inputs, **gen_kwargs)
                    gen_ids = out_ids[:, proc_inputs["input_ids"].shape[1] :]
                    decoded = self.processor.decode(gen_ids, return_format="transcription_only")
                    if isinstance(decoded, list):
                        decoded = decoded[0] if decoded else ""
                    text = str(decoded).strip()
                    if text:
                        pieces.append(text)
                transcriptions.append(" ".join(pieces).strip())

        return self.postprocess(transcriptions)


ORUK_FOURIER_2B_API_META = ModelMeta(
    name="oruk-fourier-2b",
    model_path="oruk-fourier-2b",
    description=(
        "Oruk Fourier 2B hosted English ASR API (OpenAI-compatible REST at "
        "speech-api.oruk.ai; $0.0080/min, commercial use permitted)"
    ),
    loader_class="karma.models.oruk_asr.OrukASR",
    loader_kwargs={"language": "en"},
    default_eval_kwargs={},
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    languages=["eng-Latn"],
    reference="https://speech-api.oruk.ai",
    release_date="2026-11-21",
    version="2.0",
)

ORUK_FOURIER_2B_LOCAL_META = ModelMeta(
    name=ORUK_FOURIER_MODEL_ID,
    model_path=ORUK_FOURIER_MODEL_ID,
    revision=ORUK_FOURIER_REVISION,
    description=(
        "Oruk Fourier 2B open-weight native Transformers BF16 English ASR model "
        "(Qwen3ASRForConditionalGeneration, requires transformers>=5.24.0; CC BY-NC 4.0)"
    ),
    loader_class="karma.models.oruk_asr.OrukFourierLocalASR",
    loader_kwargs={"decode_mode": "published", "language": "English"},
    default_eval_kwargs={},
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    audio_sample_rate=TARGET_SAMPLE_RATE,
    supported_audio_formats=["wav", "flac", "mp3", "m4a", "ogg", "webm"],
    languages=["eng-Latn"],
    license="CC-BY-NC-4.0",
    open_weights=True,
    reference="https://huggingface.co/oruk/oruk-fourier-2b",
    release_date="2026-11-21",
    version="2.0",
)

register_model_meta(ORUK_FOURIER_2B_API_META)
register_model_meta(ORUK_FOURIER_2B_LOCAL_META)
