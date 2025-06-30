"""
SLAKE dataset implementation with multimodal support.

This module provides the SLAKEDataset class that inherits from VQARADDataset
since they share the same structure for visual question answering.
"""

import logging

from karma.eval_datasets.vqa_rad_dataset import VQARADDataset
from karma.dataset_registry import register_dataset
from karma.eval_datasets.base_dataset import BaseMultimodalDataset

logger = logging.getLogger(__name__)

# Dataset configuration for SLAKE
DATASET_NAME = "mdwiratathya/SLAKE-vqa-english"
SPLIT = "test"
COMMIT_HASH = "8d18b4d5a4eae47101c1d9f57b99fc58df66f17e"


@register_dataset("slake", metrics=["exact_match"], task_type="vqa")
class SLAKEDataset(VQARADDataset):
    """
    SLAKE PyTorch Dataset inheriting from VQARADDataset.
    Handles visual question answering with the same structure as VQA-RAD.
    """

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        commit_hash: str = COMMIT_HASH,
        **kwargs,
    ):
        """
        Initialize SLAKE dataset.

        Args:
            **kwargs: Additional arguments passed to base class
        """
        # Override the dataset name for SLAKE

        super().__init__(
            dataset_name=dataset_name, split=split, commit_hash=commit_hash, **kwargs
        )
