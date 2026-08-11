"""
MedQAIndic dataset implementation with multimodal support.

This module provides the MedQAIndic Dataset class that implements the new
multimodal dataset interface for use with the refactored benchmark system.
"""

import logging
from typing import Any, Dict, Tuple

from karma.registries.dataset_registry import register_dataset
from karma.utils.language import normalize_language
from karma.eval_datasets.medmcqa_dataset import MedMCQADataset

logger = logging.getLogger(__name__)

# Hardcoded confinement instructions
DATASET_NAME = "ekacare/MedMCQA-Indic"
SPLIT = "test"
COMMIT_HASH = "c1e955a50a123c7534c768ec4e693b35a1817170"


@register_dataset(
    DATASET_NAME,
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    required_args=["subset"],
    metrics=["exact_match"],
    task_type="mcqa",
    default_args={"user_prompt": "medmcqa_indic_cot"},
)
class MedMCQAIndicDataset(MedMCQADataset):
    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        subset: str = "as",
        commit_hash: str = COMMIT_HASH,
        **kwargs,
    ):
        """
        Initialize MedMCQA dataset.

        Args:
            **kwargs: Additional arguments passed to base class
        """
        self.subset = normalize_language(subset)
        self.dataset_subset = self.subset
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            config=self.subset,
            commit_hash=commit_hash,
            **kwargs,
        )

    def format_item(self, sample: Dict[str, Any], **kwargs):
        return super().format_item(sample=sample, subset=self.subset)

    def extract_prediction(self, response: str, **kwargs) -> Tuple[str, bool]:
        return super().extract_prediction(response=response, subset=self.subset)
