"""
MedXpertQA MM dataset implementation with multimodal support.

This module provides the MedXpertQADataset class that implements the new
multimodal dataset interface for medical question answering with images.
"""

import logging
from typing import Dict, Any, Tuple
from datasets import Image
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.registries.dataset_registry import register_dataset
from karma.eval_datasets.base_dataset import BaseMultimodalDataset

logger = logging.getLogger(__name__)
CONFINEMENT_INSTRUCTIONS = "Output format: 'ANSWER: <option>', examples: ['ANSWER: A', 'ANSWER: B', 'ANSWER: C', 'ANSWER: D', 'ANSWER: E']"
DATASET_NAME = "ChuGyouk/MedXpertQA"
SPLIT = "test"
CONFIG = "MM"
COMMIT_HASH = "7186bd593752a47d6bd72ccf99ca67df69be18bd"


@register_dataset(
    DATASET_NAME,
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    metrics=["exact_match"],
    task_type="mcqa",
)
class MedXpertQADataset(BaseMultimodalDataset):
    """
    MedXpertQA MM PyTorch Dataset implementing the new multimodal interface.
    Handles medical question answering with images.
    """

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        config: str = CONFIG,
        **kwargs,
    ):
        """
        Initialize MedXpertQA MM dataset.

        Args:
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(
            dataset_name=dataset_name,
            config=config,
            **kwargs,
        )
        self.dataset = self.dataset.cast_column("images", [Image(decode=False)])

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        """
        Format a sample into a medical QA prompt.

        Args:
            sample: A single sample from the dataset

        Returns:
            Dictionary with formatted prompt and expected output
        """
        question = sample.get("question", "")
        options = sample.get("options", {})
        label = sample.get("label", "")
        images = [image["bytes"] for image in sample["images"]]

        formatted_choices = []
        for key, value in options.items():
            formatted_choices.append(f"{key}. {value}")

        # Create medical QA prompt
        choices_text = "\n".join(formatted_choices)
        prompt = f"Question: {question}\n\n{choices_text}\n\n{CONFINEMENT_INSTRUCTIONS}"

        processed_sample = DataLoaderIterable(
            input=prompt,
            expected_output=label,
            images=images,  # Include images for multimodal models
        )

        return processed_sample

    def extract_prediction(self, response: str) -> Tuple[str, bool]:
        """
        Extract the answer from model response.

        Args:
            response: Model's response text

        Returns:
            Extracted answer letter (A, B, C, D, or E)
        """
        response = response.strip().upper()

        # Look for single letter answers
        valid_answers = ["A", "B", "C", "D", "E"]

        # Check if response is just a single letter
        if response in valid_answers:
            return response, True

        # Look for "Answer: X" pattern
        if "ANSWER:" in response:
            answer_part = (
                response.split("ANSWER:")[-1].replace(")", "").replace("(", "").strip()
            )
            if answer_part and answer_part[0] in valid_answers:
                return answer_part[0], True

        # Look for first occurrence of valid answer
        for char in response.split(" "):
            if char in valid_answers:
                return char, True

        # Default to A if no valid answer found
        return response, False
