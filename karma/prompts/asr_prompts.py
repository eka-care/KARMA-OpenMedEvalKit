"""User-prompt (task instruction) bodies for speech-to-text / ASR datasets.

General multimodal models (gemini, openai audio) consume this as the
transcription instruction; dedicated ASR models transcribe regardless and
ignore it.
"""

from karma.prompts import Prompt

ASR_TRANSCRIBE = Prompt(
    name="asr_transcribe",
    version=1,
    role="user",
    body="Transcribe the given audio. Provide only the text without timestamps or speaker information.",
)
