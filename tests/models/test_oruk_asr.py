"""Unit tests for Oruk Fourier 2B hosted API and local Transformers adapters."""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.models.oruk_asr import (
    ORUK_FOURIER_2B_API_META,
    ORUK_FOURIER_2B_LOCAL_META,
    ORUK_FOURIER_MODEL_ID,
    ORUK_FOURIER_REVISION,
    OrukASR,
    OrukFourierLocalASR,
    _detect_audio_format,
    _split_chunks,
    _validate_base_url,
)


def test_model_meta_registration_and_pinning():
    assert ORUK_FOURIER_2B_API_META.name == "oruk-fourier-2b"
    assert ORUK_FOURIER_2B_API_META.loader_class == "karma.models.oruk_asr.OrukASR"
    assert ORUK_FOURIER_2B_LOCAL_META.name == ORUK_FOURIER_MODEL_ID
    assert ORUK_FOURIER_2B_LOCAL_META.revision == ORUK_FOURIER_REVISION
    assert ORUK_FOURIER_2B_LOCAL_META.license == "CC-BY-NC-4.0"


def test_detect_audio_format_and_base_url_validation():
    assert _detect_audio_format(b"RIFF1234WAVEfmt ") == "wav"
    assert _detect_audio_format(b"fLaC\x00\x00\x00\x22xxxx") == "flac"
    assert _detect_audio_format(b"\x00\x00\x00\x18ftypM4A ") == "m4a"
    assert _detect_audio_format(b"OggS\x00\x02\x00\x00xxxx") == "ogg"
    assert _validate_base_url("https://speech-api.oruk.ai/v1/") == "https://speech-api.oruk.ai/v1"
    assert _validate_base_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"
    with pytest.raises(ValueError):
        _validate_base_url("http://example.com/v1")


def test_oruk_hosted_asr_requires_key(monkeypatch):
    monkeypatch.delenv("ORUK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ORUK_API_KEY"):
        OrukASR()
    with pytest.raises(ValueError, match="Unsupported Oruk Speech API language"):
        OrukASR(api_key="test_key", language="fr")


def test_oruk_hosted_asr_transcribe_and_redacts_key():
    captured = {}

    class _Transcriptions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(text="  Take azithromycin twice daily.  ")

    m = OrukASR.__new__(OrukASR)
    m.model_id = "oruk-fourier-2b"
    m.api_key = "oruk_secret_test_key_12345"
    m.base_url = "https://speech-api.oruk.ai/v1"
    m.language = "en"
    m.is_loaded = True
    m.client = types.SimpleNamespace(audio=types.SimpleNamespace(transcriptions=_Transcriptions()))

    out = m.run_with_metadata([DataLoaderIterable(audio=b"RIFF1234WAVEfmt ")])
    assert out == [{"text": "Take azithromycin twice daily.", "tool_trace": "", "error": None}]
    assert captured["model"] == "oruk-fourier-2b"
    assert captured["language"] == "en"
    assert "prompt" not in captured
    assert captured["file"].name == "audio.wav"

    # Key redaction on error
    class _FailingTranscriptions:
        def create(self, **kwargs):
            raise RuntimeError("401 unauthorized for key oruk_secret_test_key_12345")

    m.client = types.SimpleNamespace(audio=types.SimpleNamespace(transcriptions=_FailingTranscriptions()))
    err_out = m.run_with_metadata([DataLoaderIterable(audio=b"RIFF1234WAVEfmt ")])
    assert err_out[0]["text"] == ""
    assert "oruk_secret_test_key_12345" not in (err_out[0]["error"] or "")
    assert "[REDACTED]" in (err_out[0]["error"] or "")


def test_oruk_local_asr_decode_modes_and_chunking():
    local_pub = OrukFourierLocalASR(decode_mode="published", device="cpu")
    assert local_pub._generation_kwargs() == {}

    local_greedy = OrukFourierLocalASR(decode_mode="greedy", max_new_tokens=256, device="cpu")
    assert local_greedy._generation_kwargs() == {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 256,
    }

    with pytest.raises(ValueError):
        OrukFourierLocalASR(decode_mode="invalid", device="cpu")

    long_wav = np.zeros(16_000 * 65, dtype=np.float32)
    chunks = _split_chunks(long_wav, chunk_seconds=30.0)
    assert len(chunks) == 3
    assert chunks[0].size == 16_000 * 30
    assert chunks[2].size == 16_000 * 5


def test_oruk_local_asr_passes_pinned_revision_for_green_hf_attribution():
    fake_proc = MagicMock()
    fake_proc.tokenizer = MagicMock()
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model

    fake_tf = types.SimpleNamespace(
        AutoProcessor=MagicMock(from_pretrained=MagicMock(return_value=fake_proc)),
        AutoModelForMultimodalLM=MagicMock(from_pretrained=MagicMock(return_value=fake_model)),
    )
    local = OrukFourierLocalASR(device="cpu")
    with patch.dict("sys.modules", {"transformers": fake_tf}):
        local.load_model()

    fake_tf.AutoProcessor.from_pretrained.assert_called_once_with(
        ORUK_FOURIER_MODEL_ID, revision=ORUK_FOURIER_REVISION
    )
    call_args, call_kwargs = fake_tf.AutoModelForMultimodalLM.from_pretrained.call_args
    assert call_args == (ORUK_FOURIER_MODEL_ID,)
    assert call_kwargs["revision"] == ORUK_FOURIER_REVISION
