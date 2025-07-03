"""
MedMCQA dataset implementation with multimodal support.

This module provides the MedMCQADataset class that implements the new
multimodal dataset interface for use with the refactored benchmark system.
"""

import logging
from typing import Dict, Any, Tuple

from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset

logger = logging.getLogger(__name__)

# Hardcoded confinement instructions
CONFINEMENT_INSTRUCTIONS = "Output format: 'ANSWER: <option>', examples: ['ANSWER: A', 'ANSWER: B', 'ANSWER: C', 'ANSWER: D']"
DATASET_NAME = "openlifescienceai/medmcqa"
SPLIT = "validation"
COMMIT_HASH = "91c6572c454088bf71b679ad90aa8dffcd0d5868"


@register_dataset(
    DATASET_NAME,
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    metrics=["exact_match"],
    task_type="mcqa",
)
class MedMCQADataset(BaseMultimodalDataset):
    """
    MedMCQA PyTorch Dataset implementing the new multimodal interface.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialize MedMCQA dataset.

        Args:
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(**kwargs)

    def format_item(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a sample into a text prompt for MedMCQA.

        Args:
            sample: A single sample from the dataset

        Returns:
            Formatted text prompt
        """
        input_text = self._format_question(sample)

        # Parse correct answer from cop field
        cop = sample["cop"]
        choice_labels = ["A", "B", "C", "D"]
        correct_answer = choice_labels[cop]
        prompt = f"{input_text}\n\n{CONFINEMENT_INSTRUCTIONS}"

        processed_sample = {
            "input": prompt,
            "expected_output": correct_answer,  # Change it to Label
        }
        # Add confinement instructions to the question and options

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
        Format a single MedMCQA question.

        Args:
            data: Dictionary containing question data with keys:
                - question: The question text
                - opa, opb, opc, opd: The four answer choices
                - cop: The correct answer (if include_answer is True)
            include_answer: Whether to include the answer in the formatted question

        Returns:
            Formatted question string
        """
        question = data["question"]
        choices = [data["opa"], data["opb"], data["opc"], data["opd"]]

        # Format choices as A, B, C, D
        formatted_choices = []
        choice_labels = ["A", "B", "C", "D"]
        for i, choice in enumerate(choices):
            formatted_choices.append(f"{choice_labels[i]}. {choice}")

        formatted_question = f"{question}\n" + "\n".join(formatted_choices)

        return formatted_question
