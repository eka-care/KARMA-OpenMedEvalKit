"""
MedQA dataset implementation with multimodal support.

This module provides the MedQADataset class that implements the new
multimodal dataset interface for use with the refactored benchmark system.
"""

import json
import logging
from typing import Any, Dict, Tuple

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset

logger = logging.getLogger(__name__)

# Hardcoded confinement instructions
CONFINEMENT_INSTRUCTIONS = """
You are an expert at Indian Pharmacology.
Solve them in a step-by-step fashion.
Output only the final answer (e.g., A, B, C, D, or E)
Question: <QUESTION>
"""
DATASET_NAME = "ekacare/Eka_NFI_MCQA"
SPLIT = "test"
COMMIT_HASH = "428cf44c302baf9e649c257ce07d7fb718c118dd"


@register_dataset(
    DATASET_NAME,
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    metrics=["exact_match"],
    task_type="mcqa",
)
class EkaNFIValidationSetDataset(BaseMultimodalDataset):
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
        Initialize EkaNFIValidationSet dataset.

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
        Format a sample into a text prompt for EkaMedicationSFTValidationSet.

        Args:
            sample: A single sample from the dataset

        Returns:
            Dictionary with formatted prompt and expected output
        """
        question = self._format_question(sample)

        # Parse correct answer from Correct Option field
        correct_option = sample["answer"]
        if correct_option in ["true", "false"]:
            correct_option = "A" if correct_option == "true" else "B"
        prompt = self.confinement_instructions.replace("<QUESTION>", question)
        processed_sample = DataLoaderIterable(
            input=prompt,
            expected_output=correct_option,
        )

        return processed_sample

    def extract_prediction(self, response: str) -> Tuple[str, bool]:
        answer, success = "", False
        # Try "Final Answer:" format first
        if "Final Answer:" in response:
            raw = response.split("Final Answer:")[1].strip()
        else:
            raw = response.strip()
        # Extract first valid letter (A-E), handles: "A", "A**", "**A**", "(A)", "A.", etc.
        raw = raw.strip("()*")
        if raw and raw[0].upper() in "ABCDE":
            answer = raw[0].upper()
            success = True
        if not success:
            logger.warning(f"No answer found in response: {response}")
        return answer, success

    def _format_question(self, data: Dict[str, Any]) -> str:
        """
        Format a single EkaMedicationSFTValidationSet question.

        Args:
            data: Dictionary containing question data with keys:
                - question: The question text
                - options: Dict with A/B/C/D options

        Returns:
            Formatted question string
        """
        question = data["question"]
        options = data["options"]
        # Format choices as A, B, C, D
        formatted_choices = []
        if options:
            for label, option in json.loads(options).items():
                formatted_choices.append(f"{label}. {option}")
        else:
            formatted_choices = ["A. True", "B. False"]

        formatted_question = (
            f"{question}\n" + "\n".join(formatted_choices) + "\nFinal Answer:"
        )

        return formatted_question
