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
CONFINEMENT_INSTRUCTIONS = """<QUESTION> Think
step by step through each of the multiple choice options. You MUST end your response with "Final
Answer:" followed by only the letter corresponding to the correct answer enclosed in parentheses)."""
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
        split: str = SPLIT,
        commit_hash: str = COMMIT_HASH,
        config: str = CONFIG,
        confinement_instructions: str = CONFINEMENT_INSTRUCTIONS,
        **kwargs,
    ):
        """
        Initialize MedXpertQA MM dataset.

        Args:
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            commit_hash=commit_hash,
            config=config,
            confinement_instructions=confinement_instructions,
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
        prompt = self.confinement_instructions.replace("<QUESTION>", question+"\n\n"+choices_text)
        processed_sample = DataLoaderIterable(
            input=prompt,
            expected_output=label,
            images=images,  # Include images for multimodal models
        )

        return processed_sample

    def extract_prediction(self, response: str) -> Tuple[str, bool]:
        answer, success = "", False
        if "Final Answer:" in response:
            answer = response.split("Final Answer:")[1].strip()
            if answer.startswith('(') and answer.endswith(')'):
                answer = answer[1:-1]
            success = True
        if not answer:
            logger.warning(f"No answer found in response: {response}")
        return answer, success