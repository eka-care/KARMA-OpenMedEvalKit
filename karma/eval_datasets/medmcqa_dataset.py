"""
MedMCQA dataset implementation with multimodal support.

This module provides the MedMCQADataset class that implements the new
multimodal dataset interface for use with the refactored benchmark system.
"""

import logging
from typing import Dict, Any, Tuple

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
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        commit_hash: str = COMMIT_HASH,
        **kwargs,
    ):
        """
        Initialize MedMCQA dataset.

        Args:
            **kwargs: Additional arguments passed to base class
        """
        # Confinement instructions
        self.confinement_instructions = CONFINEMENT_INSTRUCTIONS
        super().__init__(dataset_name=dataset_name, split=split, commit_hash=commit_hash, **kwargs)

    def format_item(self, sample: Dict[str, Any], **kwargs):
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
        prompt = self.confinement_instructions.replace("<QUESTION>", input_text)
        processed_sample = DataLoaderIterable(
            input=prompt, expected_output=correct_answer, **kwargs
        )

        # Add confinement instructions to the question and options

        return processed_sample

    def extract_prediction(self, response: str, **kwargs) -> Tuple[str, bool]:
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
