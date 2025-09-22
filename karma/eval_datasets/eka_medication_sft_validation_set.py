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
CONFINEMENT_INSTRUCTIONS = """Instructions: The following are multiple choice questions about medical knowledge. Solve them in a
step-by-step fashion, starting by summarizing the available information. Output correct options from the
given options as the final answer. Question: <QUESTION> Response (think step by step and then
end with "Final Answer:" followed by correct option or options corresponding to the correct answers enclosed in
parentheses like (A, B, C, D), (B), (C, D), etc.)"""
DATASET_NAME = "ekacare/eka_medication_sft_validation_set"
SPLIT = "test"
COMMIT_HASH = "9b4853b7c59b82f66bd49cf4afdc467f97aa900a"


@register_dataset(
    DATASET_NAME,
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    metrics=["exact_match"],
    task_type="mcqa",
    required_args=["subset"],
    default_args={"subset": "common_drugs"},
)
class EkaMedicationSFTValidationSetDataset(BaseMultimodalDataset):
    """
    EkaMedicationSFTValidationSet PyTorch Dataset implementing the new multimodal interface.
    """

    def __init__(
        self,
        dataset_name=DATASET_NAME,
        split=SPLIT,
        subset: str = "common_drugs",
        confinement_instructions: str = CONFINEMENT_INSTRUCTIONS,
        commit_hash: str = COMMIT_HASH,
        **kwargs,
    ):
        """
        Initialize EkaMedicationSFTValidationSet dataset.

        Args:
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            confinement_instructions=confinement_instructions,
            config=subset,
            commit_hash=commit_hash,
            **kwargs,
        )
        self.subset = subset
        self.dataset_name = f"{DATASET_NAME}-{self.subset}"

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
        if correct_option in ["True", "False"]:
            correct_option = "A" if correct_option == "True" else "B"
        prompt = self.confinement_instructions.replace("<QUESTION>", question)
        processed_sample = DataLoaderIterable(
            input=prompt,
            expected_output=correct_option,
        )

        return processed_sample

    def extract_prediction(self, response: str) -> Tuple[str, bool]:
        answer, success = "", False
        if "Final Answer:" in response:
            answer = response.split("Final Answer:")[1].strip()
            print(f"Answer: {answer}")
            # Remove parentheses if present
            # if answer.startswith("(") and answer.endswith(")"):
            # answer = answer[1:-1]
            correct_options = []
            for option in ["A", "B", "C", "D", "E"]:
                if option in answer:
                    correct_options.append(option)
            answer = ", ".join(correct_options)
            success = True
        if not answer:
            logger.warning(f"No answer found in response: {response}")
        print(f"Answer: {answer}, Success: {success}")
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
        
        formatted_question = f"{question}\n" + "\n".join(formatted_choices)

        return formatted_question
