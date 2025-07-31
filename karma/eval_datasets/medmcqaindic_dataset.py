"""
MedQAIndic dataset implementation with multimodal support.

This module provides the MedQAIndic Dataset class that implements the new
multimodal dataset interface for use with the refactored benchmark system.
"""

import logging
from typing import Any, Dict, Tuple

from karma.registries.dataset_registry import register_dataset
from karma.eval_datasets.medmcqa_dataset import MedMCQADataset

logger = logging.getLogger(__name__)

# Hardcoded confinement instructions
DATASET_NAME = "ekacare/MedMCQA-Indic"
SPLIT = "test"
COMMIT_HASH = "dc18742d78a3486eef3d68b610ec47411ae383dd"
CONFINEMENT_INSTRUCTIONS = """Instructions: The following are multiple choice questions about medical knowledge. Solve them in a
step-by-step fashion, starting by summarizing the available information. Output a single option from the
four options as the final answer. Question: <QUESTION> Response (think step by step and then
end with "Final Answer:" followed by *only* the letter corresponding to the correct answer enclosed in
parentheses)"""


@register_dataset(
    DATASET_NAME,
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    required_args=["subset"],
    metrics=["exact_match"],
    task_type="mcqa",
    optional_args=["confinement_instructions"],
)
class MedMCQAIndicDataset(MedMCQADataset):
    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        subset: str = "as",
        commit_hash: str = COMMIT_HASH,
        confinement_instructions: str = CONFINEMENT_INSTRUCTIONS,
        **kwargs,
    ):
        """
        Initialize MedMCQA dataset.

        Args:
            **kwargs: Additional arguments passed to base class
        """
        self.subset = subset
        self.dataset_name = f"{DATASET_NAME}-{self.subset}"
        # kwargs.pop('dataset_name', None)
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            config=self.subset,
            commit_hash=commit_hash,
            confinement_instructions=confinement_instructions,
            **kwargs,
        )

    def format_item(self, sample: Dict[str, Any], **kwargs):
        return super().format_item(sample=sample, subset=self.subset)

    def extract_prediction(self, response: str, **kwargs) -> Tuple[str, bool]:
        return super().extract_prediction(response=response, subset=self.subset)
