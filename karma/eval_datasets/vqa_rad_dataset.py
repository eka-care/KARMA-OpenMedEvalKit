"""
VQA-RAD dataset implementation with multimodal support.

This module provides the VQARADDataset class that implements the new
multimodal dataset interface for visual question answering on radiology images.
"""

import logging
from typing import Dict, Any
from datasets import Image
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset
from karma.data_models.dataloader_iterable import DataLoaderIterable

logger = logging.getLogger(__name__)

# Hardcoded confinement instructions for VQA
DATASET_NAME = "flaviagiammarino/vqa-rad"
SPLIT = "test"
COMMIT_HASH = "bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9"
CONFINEMENT_INSTRUCTIONS = """Given this radiology image, which can be a frontal chest X-ray, a single slice head or
abdominal CT or MR image, provide a very short, definitive, and concise answer (if possible, a single
word) to the following question: <QUESTION>"""


@register_dataset(
    DATASET_NAME,
    split=SPLIT,
    commit_hash=COMMIT_HASH,
    optional_args=["confinement_instructions"],
    metrics=["exact_match", "tokenised_f1"],
    task_type="vqa",
)
class VQARADDataset(BaseMultimodalDataset):
    """
    VQA-RAD PyTorch Dataset implementing the new multimodal interface.
    Handles visual question answering for radiology images.
    """

    def __init__(
        self,
        dataset_name=DATASET_NAME,
        split=SPLIT,
        commit_hash=COMMIT_HASH,
        confinement_instructions: str = CONFINEMENT_INSTRUCTIONS,
        **kwargs,
    ):
        """
        Initialize VQA-RAD dataset.

        Args:
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            commit_hash=commit_hash,
            confinement_instructions=confinement_instructions,
            **kwargs,
        )
        self.dataset = self.dataset.cast_column("image", Image(decode=False))

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        """
        Format a sample into a VQA prompt.

        Args:
            sample: A single sample from the dataset

        Returns:
            Dictionary with formatted prompt and expected output
        """
        question = sample.get("question", "")
        answer = sample.get("answer", "").lower()
        image = sample["image"]["bytes"]

        # Create VQA prompt
        prompt = self.confinement_instructions.replace("<QUESTION>", question)

        processed_sample = DataLoaderIterable(
            input=prompt,
            expected_output=answer,
            images=[image],  # Include image for multimodal models
        )

        return processed_sample
