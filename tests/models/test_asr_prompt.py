"""ASR prompt unification: general ASR models use the dataset user_prompt as the
transcription instruction, and error (never silently assume a task) if it's absent."""

import types

import pytest

from karma.models.openai_asr import OpenAIASR
from karma.models.gemini_asr import GeminiASR
from karma.prompts import resolve_prompt

REGISTRY_DEFAULT = "Transcribe the given audio. Provide only the text without timestamps or speaker information."


class _CapturingTranscriptions:
    def __init__(self):
        self.prompt = None

    def create(self, model, file, prompt):
        self.prompt = prompt
        return types.SimpleNamespace(text="ok")


def _openai(monkeypatch):
    m = OpenAIASR.__new__(OpenAIASR)
    m.model_id = "whisper-1"
    monkeypatch.setattr(m, "detect_audio_format", lambda b: "wav")
    cap = _CapturingTranscriptions()
    m.client = types.SimpleNamespace(audio=types.SimpleNamespace(transcriptions=cap))
    return m, cap


def _gemini():
    m = GeminiASR.__new__(GeminiASR)
    m.model_id = "gemini-x"
    captured = {}
    m.client = types.SimpleNamespace(
        files=types.SimpleNamespace(upload=lambda file: "FILE"),
        models=types.SimpleNamespace(
            generate_content=lambda model, config, contents: (
                captured.__setitem__("instruction", contents[0]),
                types.SimpleNamespace(text="ok"),
            )[1]
        ),
    )
    m._build_generation_config = lambda *a, **k: None
    return m, captured


def test_registry_default_matches_asr_transcribe():
    assert resolve_prompt("asr_transcribe").body == REGISTRY_DEFAULT


def test_openai_uses_dataset_instruction(monkeypatch):
    m, cap = _openai(monkeypatch)
    m.transcribe(b"RIFFxxxx", "Diarize this audio: who spoke when.")
    assert cap.prompt == "Diarize this audio: who spoke when."


def test_openai_errors_without_instruction(monkeypatch):
    m, _ = _openai(monkeypatch)
    with pytest.raises(ValueError):
        m.transcribe(b"RIFFxxxx", None)


def test_gemini_uses_dataset_instruction():
    m, captured = _gemini()
    m.transcribe(b"audio", "Summarize the audio.")
    assert captured["instruction"] == "Summarize the audio."


def test_gemini_errors_without_instruction():
    m, _ = _gemini()
    with pytest.raises(ValueError):
        m.transcribe(b"audio", None)
