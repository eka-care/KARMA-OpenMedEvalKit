"""
IN22Conv dataset implementation with multimodal support.

This module provides the IN22ConvDataset class that implements the new
multimodal dataset interface for translation from English to Indian languages.
"""

from typing import Dict, Any, Tuple, List, Optional

from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset
from karma.data_models.dataloader_iterable import DataLoaderIterable

CONFINEMENT_INSTRUCTIONS = "Translate the given text to the target language. Output only the translation without any additional text."
DATASET_NAME = "ai4bharat/IN22-Conv"
SPLIT = "test"
COMMIT_HASH = "18cd45870ff0a9e65df9b80dbbcc615eec0e4899"
ID_TO_CODE = {
    "en": "eng_Latn",
    "as": "asm_Beng",
    "bn": "ben_Beng",
    "brx": "brx_Deva",
    "doi": "doi_Deva",
    "gu": "guj_Gujr",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "kok": "gom_Deva",
    "ks": "kas_Arab",
    "mai": "mai_Deva",
    "ml": "mal_Mlym",
    "mni": "mni_Mtei",
    "mr": "mar_Deva",
    "ne": "npi_Deva",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "sa": "san_Deva",
    "sat": "sat_Olck",
    "sd": "snd_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ur": "urd_Arab",
}
CODE_TO_NAME = {
    "eng_Latn": "English",
    "asm_Beng": "Assamese",
    "ben_Beng": "Bengali",
    "brx_Deva": "Bodo",
    "doi_Deva": "Dogri",
    "guj_Gujr": "Gujarati",
    "hin_Deva": "Hindi",
    "kan_Knda": "Kannada",
    "gom_Deva": "Konkani",
    "kas_Arab": "Kashmiri",
    "mai_Deva": "Maithili",
    "mal_Mlym": "Malayalam",
    "mni_Mtei": "Manipuri",
    "mar_Deva": "Marathi",
    "npi_Deva": "Nepali",
    "ory_Orya": "Odia",
    "pan_Guru": "Punjabi",
    "san_Deva": "Sanskrit",
    "sat_Olck": "Santali",
    "snd_Deva": "Sindhi",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "urd_Arab": "Urdu",
}


@register_dataset(
    dataset_name=DATASET_NAME,
    metrics=["bleu"],
    task_type="translation",
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    processors=["devnagari_transliterator"],
    required_args=["source_language", "target_language"],
    optional_args=["domain", "processors", "confinement_instructions"],
    default_args={"source_language": "en", "domain": "conversational"},
)
class IN22ConvDataset(BaseMultimodalDataset):
    """
    IN22Conv PyTorch Dataset implementing the new multimodal interface.
    Translates from English to specified Indian language.
    """

    def __init__(
        self,
        source_language: str,
        target_language: str,
        dataset_name=DATASET_NAME,
        split=SPLIT,
        domain: str = "conversational",
        processors: Optional[List] = None,
        confinement_instructions: Optional[str] = CONFINEMENT_INSTRUCTIONS,
        **kwargs,
    ):
        """
        Initialize IN22Conv dataset.

        Args:
            source_language: Source language code (e.g., 'en')
            target_language: Target language code (e.g., 'hi')
            domain: Domain type (e.g., 'conversational')
            processors: List of processor instances (passed by orchestrator)
            dataset_name: Name of the HuggingFace dataset
            split: Dataset split to use
            commit_hash: Specific commit hash of the dataset
            **kwargs: Additional arguments passed to parent class
        """
        # Validate language codes
        if source_language not in ID_TO_CODE:
            raise ValueError(
                f"Unsupported source language: {source_language}. "
                f"Supported languages: {list(ID_TO_CODE.keys())}"
            )
        if target_language not in ID_TO_CODE:
            raise ValueError(
                f"Unsupported target language: {target_language}. "
                f"Supported languages: {list(ID_TO_CODE.keys())}"
            )

        # Store language information
        self.source_language_code = source_language
        self.target_language_code = target_language
        self.source_language = ID_TO_CODE[source_language]
        self.target_language = ID_TO_CODE[target_language]
        self.domain = domain
        super().__init__(
            processors=processors,
            confinement_instructions=confinement_instructions,
            **kwargs,
        )
        self.dataset_name = (
            f"{DATASET_NAME}-{self.source_language}-{self.target_language}"
        )

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        """
        Format a sample into a translation prompt.

        Args:
            sample: A single sample from the dataset

        Returns:
            Dictionary with formatted prompt and expected output
        """
        source_text = sample.get(self.source_language, "")
        target_text = sample.get(self.target_language, "")

        # Create translation prompt
        prompt = f"Translate the following English text to {CODE_TO_NAME[self.target_language]}:\n\n{source_text}\n\n{self.confinement_instructions}"

        processed_sample = DataLoaderIterable(
            input=prompt,
            expected_output=target_text,
        )

        return processed_sample

    def extract_prediction(self, response: str) -> Tuple[str, bool]:
        """
        Extract the translation from model response.

        Args:
            response: Model's response text

        Returns:
            Extracted translation text
        """
        # For translation tasks, we typically take the response as-is
        # Remove any leading/trailing whitespace
        response = response.strip()

        # Remove common prefixes that models might add
        prefixes_to_remove = [
            "Translation:",
            "The translation is:",
            f"In {CODE_TO_NAME[self.target_language]}:",
            f"{CODE_TO_NAME[self.target_language]} translation:",
        ]

        response_lower = response.lower()
        for prefix in prefixes_to_remove:
            if response_lower.startswith(prefix.lower()):
                response = response[len(prefix) :].strip()
                break

        return response, True
