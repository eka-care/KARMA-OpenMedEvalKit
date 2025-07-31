"""
MedQA dataset implementation with multimodal support.

This module provides the MedQADataset class that implements the new
multimodal dataset interface for use with the refactored benchmark system.
"""

import logging
from typing import Any, Dict, Tuple

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset

logger = logging.getLogger(__name__)

# Hardcoded confinement instructions
CONFINEMENT_INSTRUCTIONS = """Instructions: The following are multiple choice questions about medical knowledge. Solve them in a
step-by-step fashion, starting by summarizing the available information. Output a single option from the
four options as the final answer. Question: <QUESTION> Response (think step by step and then
end with "Final Answer:" followed by *only* the letter corresponding to the correct answer enclosed in
parentheses)"""
DATASET_NAME = "openlifescienceai/medqa"
SPLIT = "test"
COMMIT_HASH = "153e61cdd129eb79d3c27f82cdf3bc5e018c11b0"


@register_dataset(
    DATASET_NAME,
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    metrics=["exact_match"],
    task_type="mcqa",
)
class MedQADataset(BaseMultimodalDataset):
    """
    MedQA PyTorch Dataset implementing the new multimodal interface.
    """

    def __init__(
        self,
        dataset_name=DATASET_NAME,
        split=SPLIT,
        confinement_instructions: str = CONFINEMENT_INSTRUCTIONS,
        commit_hash: str = COMMIT_HASH,
        **kwargs,
    ):
        """
        Initialize MedQA dataset.

        Args:
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            confinement_instructions=confinement_instructions,
            commit_hash=commit_hash,
            **kwargs,
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
        prompt = self.confinement_instructions.replace("<QUESTION>", input_text)
        processed_sample = DataLoaderIterable(
            input=prompt,
            expected_output=correct_option,
        )

        return processed_sample

    def extract_prediction(self, response: str) -> Tuple[str, bool]:
        answer, success = "", False
        if "Final Answer:" in response:
            answer = response.split("Final Answer:")[1].strip()
            # Remove parentheses if present
            if answer.startswith("(") and answer.endswith(")"):
                answer = answer[1:-1]
            success = True
        if not answer:
            logger.warning(f"No answer found in response: {response}")
        return answer, success

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
            formatted_question = f"{question}\n" + "\n".join(formatted_choices)

        return formatted_question
