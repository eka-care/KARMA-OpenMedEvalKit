"""
MedXpertQA Text dataset implementation.

This module provides the MedXpertQATextDataset class for the Text config of the
TsinghuaC3I/MedXpertQA dataset — text-only medical question answering
(2450 questions, 10 options A-J). From ICML 2025 (arXiv: 2501.18362).

Sibling to ``medxpertqa_dataset.py`` (ChuGyouk/MedXpertQA MM), which is kept
untouched for reproducibility of prior runs. Both classes coexist in the
dataset registry under distinct dataset IDs.
"""

import logging
from typing import Dict, Any, Tuple

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.registries.dataset_registry import register_dataset
from karma.eval_datasets.base_dataset import BaseMultimodalDataset

logger = logging.getLogger(__name__)

CONFINEMENT_INSTRUCTIONS = """<QUESTION> Think
step by step through each of the multiple choice options. You MUST end your response with "Final
Answer:" followed by only the letter corresponding to the correct answer enclosed in parentheses)."""

DATASET_NAME = "TsinghuaC3I/MedXpertQA"
SPLIT = "test"
CONFIG = "Text"
COMMIT_HASH = "7e7c465a68eb2b866926bfa59c8c9d17a8daba65"


@register_dataset(
    DATASET_NAME,
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    metrics=["exact_match"],
    task_type="mcqa",
)
class MedXpertQATextDataset(BaseMultimodalDataset):
    """
    MedXpertQA Text PyTorch Dataset.

    Handles text-only medical question answering with 10 options (A-J) from
    the TsinghuaC3I/MedXpertQA dataset (Text config).
    """

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        commit_hash: str = COMMIT_HASH,
        config: str = CONFIG,
        confinement_instructions: str = CONFINEMENT_INSTRUCTIONS,
        **kwargs,
    ):
        """
        Initialize MedXpertQA Text dataset.

        Args:
            dataset_name: HuggingFace dataset name.
            split: Dataset split (default: ``test``).
            commit_hash: Pinned HF repo commit for reproducibility.
            config: HF dataset config / subset name (``Text``).
            confinement_instructions: Prompt template wrapping the question.
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            commit_hash=commit_hash,
            config=config,
            confinement_instructions=confinement_instructions,
            **kwargs,
        )

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        """
        Format a sample into a medical QA prompt.

        Args:
            sample: A single sample from the dataset.

        Returns:
            DataLoaderIterable with formatted prompt and expected output.
        """
        question = sample.get("question", "")
        options = sample.get("options", {})
        label = sample.get("label", "")

        formatted_choices = []
        for key, value in options.items():
            formatted_choices.append(f"{key}. {value}")

        choices_text = "\n".join(formatted_choices)
        prompt = self.confinement_instructions.replace(
            "<QUESTION>", question + "\n\n" + choices_text
        )
        processed_sample = DataLoaderIterable(
            input=prompt,
            expected_output=label,
        )

        return processed_sample

    def extract_prediction(self, response: str) -> Tuple[str, bool]:
        answer, success = "", False
        if "Final Answer:" in response:
            answer = response.split("Final Answer:")[1].strip()
            if answer.startswith("(") and answer.endswith(")"):
                answer = answer[1:-1]
            if answer:
                success = True
        if not answer:
            logger.warning(f"No answer found in response: {response}")
        return answer, success
