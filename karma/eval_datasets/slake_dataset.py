"""
SLAKE dataset implementation with multimodal support.

This module provides the SLAKEDataset class that inherits from VQARADDataset
since they share the same structure for visual question answering.
"""

import logging
from typing import Dict, Any
from karma.data_loader_iterable import DataLoaderIterable
from karma.eval_datasets.vqa_rad_dataset import VQARADDataset
from karma.registries.dataset_registry import register_dataset

logger = logging.getLogger(__name__)

# Dataset configuration for SLAKE
DATASET_NAME = "mdwiratathya/SLAKE-vqa-english"
SPLIT = "test"
COMMIT_HASH = "8d18b4d5a4eae47101c1d9f57b99fc58df66f17e"
CONFINEMENT_INSTRUCTIONS = "You may write out your argument before stating your final very short, definitive, and concise answer (if possible, a single word or the letter corresponding to your answer choice) X in the format 'Final Answer: X': "

@register_dataset(
    DATASET_NAME,
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    metrics=["exact_match", "tokenised_f1"],
    task_type="vqa",
)
class SLAKEDataset(VQARADDataset):
    """
    SLAKE PyTorch Dataset inheriting from VQARADDataset.
    Handles visual question answering with the same structure as VQA-RAD.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialize SLAKE dataset.

        Args:
            **kwargs: Additional arguments passed to base class
        """
        # Override the dataset name for SLAKE

        super().__init__(**kwargs)
    
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
        prompt = f"Question: {question}\n\n{CONFINEMENT_INSTRUCTIONS}"

        processed_sample = DataLoaderIterable(
            input=prompt,
            expected_output=answer,
            images=[image],  # Include image for multimodal models
        )

        return processed_sample
    
    def extract_answer(self, answer: str) -> str:
        """
        Extract the answer from the answer string.
        """
        return answer.split("Final Answer: ")[1].strip()
