"""
PubMedMCQA dataset implementation with multimodal support.

This module provides the PubMedMCQADataset class that implements the new
multimodal dataset interface for use with the refactored benchmark system.
"""

import logging

from karma.eval_datasets.medqa_dataset import MedQADataset
from karma.dataset_registry import register_dataset

logger = logging.getLogger(__name__)

# Hardcoded confinement instructions
DATASET_NAME = "openlifescienceai/pubmedqa"
SPLIT = "test"
COMMIT_HASH = "50fc41dcd0bd6eb63c18d436d854c6f9e8f3c7e2"


@register_dataset("pubmedqa", metrics=["exact_match"], task_type="mcqa")
class PubMedMCQADataset(MedQADataset):
    """
    PubMedMCQA PyTorch Dataset implementing the new multimodal interface.
    """

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        commit_hash: str = COMMIT_HASH,
        **kwargs,
    ):
        """
        Initialize PubMedMCQA dataset.

        Args:
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(
            dataset_name=dataset_name, split=split, commit_hash=commit_hash, **kwargs
        )
