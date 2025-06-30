"""
VQA-RAD dataset implementation with multimodal support.

This module provides the VQARADDataset class that implements the new
multimodal dataset interface for visual question answering on radiology images.
"""

import logging
from typing import Dict, Any

from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.dataset_registry import register_dataset
logger = logging.getLogger(__name__)

# Hardcoded confinement instructions for VQA
DATASET_NAME = "flaviagiammarino/vqa-rad"
SPLIT = "test"
COMMIT_HASH = "bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9"


@register_dataset("vqa_rad", metrics=["exact_match"], task_type="vqa")
class VQARADDataset(BaseMultimodalDataset):
    """
    VQA-RAD PyTorch Dataset implementing the new multimodal interface.
    Handles visual question answering for radiology images.
    """

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        commit_hash: str = COMMIT_HASH,
        **kwargs,
    ):
        """
        Initialize VQA-RAD dataset.

        Args:
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(
            dataset_name=dataset_name, split=split, commit_hash=commit_hash, **kwargs
        )

    def format_item(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a sample into a VQA prompt.

        Args:
            sample: A single sample from the dataset

        Returns:
            Dictionary with formatted prompt and expected output
        """
        question = sample.get("question", "")
        answer = sample.get("answer", "").lower()
        image = sample.get("image", None)

        # Create VQA prompt
        if answer in ["yes", "no"]:
            prompt = (
                f"Question: {question}\n\nPlease output 'yes' or 'no'(no extra output)"
            )
        else:
            prompt = f"Question: {question}\n\nPlease answer the question concisely."

        processed_sample = {
            "input": prompt,
            "expected_output": answer,
            "images": [image],  # Include image for multimodal models
        }

        return processed_sample
