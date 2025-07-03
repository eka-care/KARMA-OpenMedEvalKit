import re
from num2words import num2words
from jiwer import Compose, ToLowerCase, RemoveMultipleSpaces, RemovePunctuation
import unicodedata
from karma.processors.base import BaseProcessor
from karma.registries.processor_registry import register_processor




@register_processor(name="general_text_processor")
class GeneralTextProcessor(BaseProcessor):
    def __init__(
        self,
        use_num2text=False,
        use_punc=False,
        use_lowercasing=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.use_lowercasing = use_lowercasing
        self.use_num2text = use_num2text
        self.use_punc = use_punc
    
    @staticmethod
    def replace_digits_with_words(text):
        def repl(match):
            return num2words(int(match.group()))

        return re.sub(r"\b\d+\b", repl, text)


    @staticmethod
    def _ensure_str(text):
        if isinstance(text, list):
            return " ".join(str(t) for t in text)
        return str(text)

    def process(self, transcription: list[str]) -> list[str]:
        if self.use_num2text:
            transcription = self.apply_num2text(transcription)
        if not self.use_punc:
            transcription = self.remove_punctuation(transcription)
        if not self.use_lowercasing:
            transcription = self.apply_lowercasing(transcription)

        normalize = Compose([RemoveMultipleSpaces()])
        transcription = [GeneralTextProcessor._ensure_str(normalize(text)) for text in transcription]
        return transcription

    def apply_num2text(self, texts: list[str]) -> list[str]:
        texts_normalized = [GeneralTextProcessor.replace_digits_with_words(text) for text in texts]
        return texts_normalized

    def remove_punctuation(self, texts: list[str]) -> list[str]:
        normalize = Compose([RemovePunctuation()])
        texts_normalized = [GeneralTextProcessor._ensure_str(normalize(text)) for text in texts]
        return texts_normalized

    def apply_lowercasing(self, texts: list[str]) -> list[str]:
        normalize = Compose([ToLowerCase()])
        texts_normalized = [GeneralTextProcessor._ensure_str(normalize(text)) for text in texts]
        return texts_normalized
    
    def multilingual_normalization(self, texts: list[str]) -> list[str]:
        texts = [unicodedata.normalize('NFD', text) for text in texts]

