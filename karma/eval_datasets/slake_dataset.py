"""
SLAKE dataset implementation with multimodal support.

This module provides the SLAKEDataset class that inherits from VQARADDataset
since they share the same structure for visual question answering.
"""

import logging

from karma.eval_datasets.vqa_rad_dataset import VQARADDataset
from karma.registries.dataset_registry import register_dataset

logger = logging.getLogger(__name__)

# Dataset configuration for SLAKE
DATASET_NAME = "mdwiratathya/SLAKE-vqa-english"
SPLIT = "test"
COMMIT_HASH = "8d18b4d5a4eae47101c1d9f57b99fc58df66f17e"


@register_dataset(
    DATASET_NAME,
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    metrics=["exact_match"],
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

        super().__init__(
            dataset_name=dataset_name, split=split, commit_hash=commit_hash, **kwargs
        )
