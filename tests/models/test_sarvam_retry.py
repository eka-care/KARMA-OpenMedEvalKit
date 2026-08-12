"""Retry + per-sample isolation behavior for SarvamASR (mocked client)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from karma.utils.retry import call_with_retry, is_transient_error
from karma.models.sarvam_asr import SarvamASR
from karma.data_models.dataloader_iterable import DataLoaderIterable


class _ApiError(Exception):
    def __init__(self, status_code, message="boom"):
        super().__init__(message)
        self.status_code = status_code


def _rate_limited():
    return _ApiError(429, "status_code: 429, body: {'error': {'message': 'Rate limit exceeded.'}}")


# ---------- call_with_retry primitive ----------

def test_retries_transient_then_succeeds():
    fn = MagicMock(side_effect=[_rate_limited(), _rate_limited(), "ok"])
    with patch("karma.utils.retry.time.sleep") as sleep:
        assert call_with_retry(fn) == "ok"
    assert fn.call_count == 3
    assert [c.args[0] for c in sleep.call_args_list] == [10.0, 20.0]


def test_backoff_sequence_caps_at_max_delay():
    fn = MagicMock(side_effect=[_rate_limited()] * 3 + ["ok"])
    with patch("karma.utils.retry.time.sleep") as sleep:
        assert call_with_retry(fn) == "ok"
    assert [c.args[0] for c in sleep.call_args_list] == [10.0, 20.0, 30.0]


def test_non_retryable_raises_immediately():
    fn = MagicMock(side_effect=_ApiError(401, "invalid api key"))
    with patch("karma.utils.retry.time.sleep") as sleep:
        with pytest.raises(_ApiError):
            call_with_retry(fn)
    assert fn.call_count == 1
    assert not sleep.called


def test_exhaustion_raises_last_error():
    fn = MagicMock(side_effect=_rate_limited())
    with patch("karma.utils.retry.time.sleep"):
        with pytest.raises(_ApiError):
            call_with_retry(fn, max_attempts=4)
    assert fn.call_count == 4


def test_transient_detection_by_message_without_status():
    assert is_transient_error(RuntimeError("Rate limit exceeded."))
    assert is_transient_error(RuntimeError("upstream 503"))
    assert not is_transient_error(RuntimeError("invalid audio format"))


# ---------- SarvamASR behavior ----------

def _model_with(side_effects):
    model = SarvamASR(api_subscription_key="test-key")
    model.client = SimpleNamespace(
        speech_to_text=SimpleNamespace(transcribe=MagicMock(side_effect=side_effects))
    )
    model.is_loaded = True
    return model


def _sample():
    return DataLoaderIterable(audio=b"RIFF-fake-audio")


def test_throttled_sample_recovers_via_retry():
    ok = SimpleNamespace(transcript="hello world")
    model = _model_with([_rate_limited(), ok])
    with patch("karma.utils.retry.time.sleep"):
        out = model.run_with_metadata([_sample()])
    assert out == [{"text": "hello world", "tool_trace": "", "error": None}]


def test_exhausted_sample_isolated_run_continues():
    ok = SimpleNamespace(transcript="fine")
    effects = [_rate_limited()] * 4 + [ok]
    model = _model_with(effects)
    with patch("karma.utils.retry.time.sleep"):
        out = model.run_with_metadata([_sample(), _sample()])
    assert out[0]["text"] == "" and "429" in out[0]["error"]
    assert out[1] == {"text": "fine", "tool_trace": "", "error": None}
    assert model.client.speech_to_text.transcribe.call_count == 5


def test_auth_error_fails_fast_without_retry():
    model = _model_with([_ApiError(401, "invalid key")])
    with patch("karma.utils.retry.time.sleep") as sleep:
        out = model.run_with_metadata([_sample()])
    assert out[0]["text"] == "" and "invalid key" in out[0]["error"]
    assert model.client.speech_to_text.transcribe.call_count == 1
    assert not sleep.called


# ---------- ElevenLabsASR behavior ----------

def _eleven_model(side_effects):
    with patch.dict("os.environ", {"ELEVEN_LABS_API_KEY": "test"}), \
         patch("karma.models.eleven_labs.ElevenLabs"):
        from karma.models.eleven_labs import ElevenLabsASR
        model = ElevenLabsASR()
    model.client = SimpleNamespace(
        speech_to_text=SimpleNamespace(convert=MagicMock(side_effect=side_effects))
    )
    return model


def test_elevenlabs_throttled_sample_recovers():
    ok = SimpleNamespace(text="scribe says hi")
    model = _eleven_model([_rate_limited(), ok])
    with patch("karma.utils.retry.time.sleep"):
        out = model.run_with_metadata([_sample()])
    assert out == [{"text": "scribe says hi", "tool_trace": "", "error": None}]


def test_elevenlabs_exhausted_sample_isolated():
    ok = SimpleNamespace(text="fine")
    model = _eleven_model([_rate_limited()] * 4 + [ok])
    with patch("karma.utils.retry.time.sleep"):
        out = model.run_with_metadata([_sample(), _sample()])
    assert out[0]["text"] == "" and "429" in out[0]["error"]
    assert out[1]["text"] == "fine"


# ---------- GeminiASR behavior ----------

def _gemini_model(side_effects):
    with patch.dict("os.environ", {"GOOGLE_AI_API_KEY": "test"}), \
         patch("karma.models.gemini_asr.genai"):
        from karma.models.gemini_asr import GeminiASR
        model = GeminiASR()
    model.client = SimpleNamespace(
        files=SimpleNamespace(upload=MagicMock(return_value="file-handle")),
        models=SimpleNamespace(generate_content=MagicMock(side_effect=side_effects)),
    )
    return model


def _audio_sample():
    return DataLoaderIterable(audio=b"RIFF-fake", input="Transcribe the audio.")


def test_gemini_throttled_sample_recovers():
    ok = SimpleNamespace(text="gemini transcript")
    model = _gemini_model([_rate_limited(), ok])
    with patch("karma.utils.retry.time.sleep"):
        out = model.run_with_metadata([_audio_sample()])
    assert out == [{"text": "gemini transcript", "tool_trace": "", "error": None}]


def test_gemini_upload_terminated_is_retried():
    ok = SimpleNamespace(text="recovered")
    model = _gemini_model([RuntimeError("Upload has already been terminated"), ok])
    with patch("karma.utils.retry.time.sleep"):
        out = model.run_with_metadata([_audio_sample()])
    assert out[0]["text"] == "recovered" and out[0]["error"] is None


def test_gemini_exhausted_sample_isolated():
    ok = SimpleNamespace(text="fine")
    model = _gemini_model([_rate_limited()] * 4 + [ok])
    with patch("karma.utils.retry.time.sleep"):
        out = model.run_with_metadata([_audio_sample(), _audio_sample()])
    errored = [o for o in out if o["error"]]
    fine = [o for o in out if not o["error"]]
    assert len(errored) == 1 and "429" in errored[0]["error"]
    assert fine[0]["text"] == "fine"


# ---------- DeepgramASR behavior ----------

def _deepgram_model(side_effects):
    from karma.models.deepgram_asr import DeepgramASR
    with patch.object(DeepgramASR, "load_model"):
        model = DeepgramASR(api_key="test")
    ok_resp = lambda text: SimpleNamespace(
        results=SimpleNamespace(
            channels=[SimpleNamespace(alternatives=[SimpleNamespace(transcript=text)])]
        )
    )
    effects = [e if isinstance(e, Exception) else ok_resp(e) for e in side_effects]
    model.client = SimpleNamespace(
        listen=SimpleNamespace(
            v1=SimpleNamespace(
                media=SimpleNamespace(transcribe_file=MagicMock(side_effect=effects))
            )
        )
    )
    model.is_loaded = True
    return model


def test_deepgram_throttled_sample_recovers():
    model = _deepgram_model([_rate_limited(), "dg text"])
    with patch("karma.utils.retry.time.sleep"):
        out = model.run_with_metadata([_sample()])
    assert out == [{"text": "dg text", "tool_trace": "", "error": None}]


def test_deepgram_exhausted_sample_isolated():
    model = _deepgram_model([_rate_limited()] * 4 + ["fine"])
    with patch("karma.utils.retry.time.sleep"):
        out = model.run_with_metadata([_sample(), _sample()])
    errored = [o for o in out if o["error"]]
    assert len(errored) == 1 and "429" in errored[0]["error"]


# ---------- GnaniASR behavior ----------

def _gnani_model(side_effects):
    from karma.models.gnani_asr import GnaniASR
    model = GnaniASR(api_key="test", language="hi")
    model.session = SimpleNamespace(post=MagicMock(side_effect=side_effects))
    return model


def _gnani_response(status=200, transcript="gnani text"):
    resp = SimpleNamespace(
        status_code=status,
        text="",
        json=lambda: {"transcript": transcript},
    )
    resp.raise_for_status = (
        (lambda: None) if status < 400
        else MagicMock(side_effect=_ApiError(status, f"{status} Server Error"))
    )
    return resp


def test_gnani_throttled_sample_recovers():
    model = _gnani_model([_gnani_response(429), _gnani_response()])
    with patch("karma.utils.retry.time.sleep"):
        out = model.run_with_metadata([_sample()])
    assert out == [{"text": "gnani text", "tool_trace": "", "error": None}]


def test_gnani_unsupported_language_fails_fast():
    model = _gnani_model([_gnani_response()])
    out = model.run_with_metadata([DataLoaderIterable(audio=b"RIFF", language="xx")])
    assert out[0]["error"] and "not supported" in out[0]["error"]
    assert not model.session.post.called


# ---------- OpenAIASR behavior ----------

def _openai_asr_model(side_effects):
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}), \
         patch("karma.models.openai_asr.OpenAI"):
        from karma.models.openai_asr import OpenAIASR
        model = OpenAIASR()
    model.client = SimpleNamespace(
        audio=SimpleNamespace(
            transcriptions=SimpleNamespace(create=MagicMock(side_effect=side_effects))
        )
    )
    return model


def test_openai_asr_throttled_sample_recovers():
    ok = SimpleNamespace(text="gpt transcript")
    model = _openai_asr_model([_rate_limited(), ok])
    with patch("karma.utils.retry.time.sleep"):
        out = model.run_with_metadata(
            [DataLoaderIterable(audio=b"RIFF", input="Transcribe.")]
        )
    assert out == [{"text": "gpt transcript", "tool_trace": "", "error": None}]


def test_openai_asr_exhausted_sample_isolated():
    ok = SimpleNamespace(text="fine")
    model = _openai_asr_model([_rate_limited()] * 4 + [ok])
    with patch("karma.utils.retry.time.sleep"):
        out = model.run_with_metadata(
            [DataLoaderIterable(audio=b"RIFF", input="T.")] * 2
        )
    assert "429" in out[0]["error"] and out[1]["text"] == "fine"


# ---------- OpenAILLM / Bedrock error propagation ----------

def test_openai_llm_failure_sets_error_not_text():
    from karma.models.openai_llm import OpenAILLM
    model = OpenAILLM.__new__(OpenAILLM)
    model._openai_tools = None
    model.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=MagicMock(side_effect=_ApiError(401, "bad key")))
        )
    )
    out = model._make_single_call({"messages": []})
    assert out["text"] == "" and "bad key" in out["error"]
    normalized = model._postprocess_with_metadata([out])
    assert normalized[0]["error"] and normalized[0]["text"] == ""


def test_openai_llm_throttled_call_recovers():
    from karma.models.openai_llm import OpenAILLM
    model = OpenAILLM.__new__(OpenAILLM)
    model._openai_tools = None
    ok = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))]
    )
    model.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=MagicMock(side_effect=[_rate_limited(), ok]))
        )
    )
    with patch("karma.utils.retry.time.sleep"):
        out = model._make_single_call({"messages": []})
    assert out == {"text": "hello", "tool_trace": ""}
