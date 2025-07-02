"""
MedQA dataset implementation with multimodal support.

This module provides the MedQADataset class that implements the new
multimodal dataset interface for use with the refactored benchmark system.
"""

import logging
from typing import Dict, Any, Tuple

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.registries.dataset_registry import register_dataset
from karma.eval_datasets.base_dataset import BaseMultimodalDataset

logger = logging.getLogger(__name__)

# Hardcoded confinement instructions
CONFINEMENT_INSTRUCTIONS = "Output format: 'ANSWER: <option>', examples: ['ANSWER: A', 'ANSWER: B', 'ANSWER: C', 'ANSWER: D']"
DATASET_NAME = "openlifescienceai/medqa"
SPLIT = "test"
COMMIT_HASH = "153e61cdd129eb79d3c27f82cdf3bc5e018c11b0"


@register_dataset("medqa", metrics=["exact_match"], task_type="mcqa")
class MedQADataset(BaseMultimodalDataset):
    """
    MedQA PyTorch Dataset implementing the new multimodal interface.
    """

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        commit_hash: str = COMMIT_HASH,
        **kwargs,
    ):
        """
        Initialize MedQA dataset.

        Args:
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(
            dataset_name=dataset_name, split=split, commit_hash=commit_hash, **kwargs
        )

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        """
        Format a sample into a text prompt for MedQA.

        Args:
            sample: A single sample from the dataset

        Returns:
            Dictionary with formatted prompt and expected output
        """
        input_text = self._format_question(sample["data"])

        # Parse correct answer from Correct Option field
        correct_option = sample["data"]["Correct Option"]
        prompt = f"{input_text}\n\n{CONFINEMENT_INSTRUCTIONS}"

        processed_sample = DataLoaderIterable(
            input=prompt,
            expected_output=correct_option,
        )

        return processed_sample

    def extract_prediction(self, response: str) -> Tuple[str, bool]:
        """
        Extract the answer letter from model response.

        Args:
            response: Model's response text

        Returns:
            Extracted answer letter (A, B, C, or D)
        """
        response = response.strip().upper()
        # Look for single letter answers
        valid_answers = ["A", "B", "C", "D"]

        # Check if response is just a single letter
        if response in valid_answers:
            return response, True

        # Look for "Answer: X" pattern
        if "ANSWER:" in response:
            answer_part = response.split("ANSWER:")[-1].strip()
            if answer_part and answer_part[0] in valid_answers:
                return answer_part[0], True

        # Look for first occurrence of valid answer
        for char in response.split(" "):
            if char in valid_answers:
                return char, True

        # Default to A if no valid answer found
        return response, False

    def _format_question(self, data: Dict[str, Any]) -> str:
        """
        Format a single MedQA question.

        Args:
            data: Dictionary containing question data with keys:
                - Question: The question text
                - Options: Dict with A/B/C/D options

        Returns:
            Formatted question string
        """
        question = data["Question"]
        options = data["Options"]
        context = data.get("Context", "")
        # Format choices as A, B, C, D
        formatted_choices = []
        choice_labels = ["A", "B", "C", "D"]
        for label in choice_labels:
            if label in options:
                formatted_choices.append(f"{label}. {options[label]}")
        if context:
            formatted_question = (
                "Context: "
                + "\n".join(context)
                + f"\n\nQuestion: {question}\n"
                + "\n".join(formatted_choices)
            )
        else:
            formatted_question = f"Question: {question}\n" + "\n".join(
                formatted_choices
            )

        return formatted_question
