import re
from num2words import num2words
from karma.metrics.glm_processor import GLMProcessor
from jiwer import Compose, ToLowerCase, RemoveMultipleSpaces, RemovePunctuation

from karma.processors.base import BaseProcessor
from karma.registries.processor_registry import register_processor


def replace_digits_with_words(text):
    def repl(match):
        return num2words(int(match.group()))

    return re.sub(r"\b\d+\b", repl, text)


def _ensure_str(text):
    if isinstance(text, list):
        return " ".join(str(t) for t in text)
    return str(text)


@register_processor(name="asr_wer_preprocessor", required_args=["language"])
class ASRTextProcessor(BaseProcessor):
    def __init__(
        self,
        use_glm=False,
        use_num2text=False,
        use_punc=False,
        use_lowercasing=False,
        language: str = "en",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_glm = use_glm
        self.use_lowercasing = use_lowercasing
        self.use_num2text = use_num2text
        self.use_punc = use_punc
        self.language = language

    def process(self, transcription: list[str]) -> list[str]:
        if self.use_glm:
            glm_processor = GLMProcessor(glm_dir="./metrics/glm")
            transcription = glm_processor.apply(transcription, lang=self.language)
        if self.use_num2text:
            transcription = self.apply_num2text(transcription, self.language)
        if not self.use_punc:
            transcription = self.remove_punctuation(transcription)
        if not self.use_lowercasing:
            transcription = self.apply_lowercasing(transcription)

        normalize = Compose([RemoveMultipleSpaces()])
        transcription = [_ensure_str(normalize(text)) for text in transcription]
        return transcription

    def apply_glm(self, text: str, lang: str) -> str:
        # Apply global mapping rules (e.g., SCLITE GLM)
        return text

    def apply_num2text(self, texts: list[str], lang: str) -> list[str]:
        texts_normalized = [replace_digits_with_words(text) for text in texts]
        return texts_normalized

    def remove_punctuation(self, texts: list[str]) -> list[str]:
        normalize = Compose([RemovePunctuation()])
        texts_normalized = [_ensure_str(normalize(text)) for text in texts]
        return texts_normalized

    def apply_lowercasing(self, texts: list[str]) -> list[str]:
        normalize = Compose([ToLowerCase()])
        texts_normalized = [_ensure_str(normalize(text)) for text in texts]
        return texts_normalized
