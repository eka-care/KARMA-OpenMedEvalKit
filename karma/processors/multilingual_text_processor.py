import re
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict
from karma.processors.base import BaseProcessor
from karma.registries.processor_registry import register_processor
from langdetect import detect, LangDetectException
from google.transliteration import transliterate_word
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory


@register_processor(name="multilingual_text_processor", required_args=["language"])
class MultilingualTextProcessor(BaseProcessor):
    """
    A minimal GLM processor that loads glm_<lang>.txt files
    and applies word-boundary substitutions to a list of strings.
    """

    def __init__(
        self, glm_dir: str = "karma/processors/glm", language: str = "hi", **kwargs
    ):
        super().__init__(**kwargs)
        self.glm_dir = Path(glm_dir)
        self._rules_cache: Dict[str, List[Tuple[re.Pattern, str]]] = {}
        self.language = language
        self.normalizer = IndicNormalizerFactory().get_normalizer(self.language)

    def _load_rules(self) -> List[Tuple[re.Pattern, str]]:
        if self.language in self._rules_cache:
            return self._rules_cache[self.language]

        glm_path = self.glm_dir / f"glm_{self.language}.txt"
        if not glm_path.exists():
            raise FileNotFoundError(f"GLM file not found: {glm_path}")

        rules = []
        with open(glm_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                src, tgt = parts
                pattern = re.compile(rf"\b{re.escape(src)}\b")
                rules.append((pattern, tgt))

        # Sort by descending pattern length (to prefer longer matches)
        rules.sort(key=lambda r: len(r[0].pattern), reverse=True)
        self._rules_cache[self.language] = rules
        return rules

    def process(self, lines: List[str]) -> List[str]:
        rules = self._load_rules()
        for i, text in enumerate(lines):
            if text and text.strip():  # Check for non-empty and non-whitespace text
                try:
                    # Only attempt language detection if text is long enough and has content
                    if len(text.strip()) >= 3:  # Minimum length for reliable detection
                        script = detect(text)
                        if script == "en":
                            candidates = transliterate_word(text, lang_code=self.language)
                            if candidates:
                                text = candidates[0]
                except LangDetectException:
                    # If language detection fails, assume it's not English and continue processing
                    pass
                
                text = re.sub(
                    r"^[\u0900-\u0903\u093C\u093E-\u094D]+", "", text
                )  # remove nuktas, bindu, and matras in the beginning of the text
                text = unicodedata.normalize("NFC", text)
                text = self.normalizer.normalize(text)
                text = self._apply_line(text, rules)
            lines[i] = text

        return lines

    def _apply_line(self, text: str, rules: List[Tuple[re.Pattern, str]]) -> str:
        """
        Apply the rules to the text.
        """
        for pattern, repl in rules:
            text = pattern.sub(repl, text)
        return text
