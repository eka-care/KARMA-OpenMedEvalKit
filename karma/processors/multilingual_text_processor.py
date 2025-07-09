import re
from pathlib import Path
from typing import List, Tuple, Dict
from karma.processors.base import BaseProcessor
from karma.registries.processor_registry import register_processor
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
        lines = [self.normalizer.normalize(line) for line in lines]
        flat_lines = []
        for i, line in enumerate(lines):
            if isinstance(line, list):
                flat_lines.extend(line)
            elif isinstance(line, str):
                flat_lines.append(line)
            else:
                raise TypeError(
                    f"Line {i} is not a string or list: {line} ({type(line)})"
                )
        return [self._apply_line(line, rules) for line in flat_lines]

    def _apply_line(self, text: str, rules: List[Tuple[re.Pattern, str]]) -> str:
        for pattern, repl in rules:
            text = pattern.sub(repl, text)
        return text
