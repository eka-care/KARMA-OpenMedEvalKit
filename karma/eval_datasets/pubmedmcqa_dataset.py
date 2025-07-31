"""
PubMedMCQA dataset implementation with multimodal support.

This module provides the PubMedMCQADataset class that implements the new
multimodal dataset interface for use with the refactored benchmark system.
"""

import logging
from typing import Dict, Any
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.medqa_dataset import MedQADataset
from karma.registries.dataset_registry import register_dataset

logger = logging.getLogger(__name__)

# Hardcoded confinement instructions
DATASET_NAME = "openlifescienceai/pubmedqa"
SPLIT = "test"
COMMIT_HASH = "50fc41dcd0bd6eb63c18d436d854c6f9e8f3c7e2"
CONFINEMENT_INSTRUCTIONS = """Instructions: The following are multiple choice questions about medical knowledge. Solve them in a
step-by-step fashion, starting by summarizing the available information. Output a single option from the
four options as the final answer. Answer the following question given the context (reply with one of the
options): Context: <CONTEXT> Question: <QUESTION> Response (think step by step and
then end with "Final Answer:" followed by *only* the letter corresponding to the correct answer enclosed
in parentheses)"""


@register_dataset(
    DATASET_NAME,
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    metrics=["exact_match"],
    task_type="mcqa",
)
class PubMedMCQADataset(MedQADataset):
    """
    PubMedMCQA PyTorch Dataset implementing the new multimodal interface.
    """

    def __init__(
        self,
        dataset_name=DATASET_NAME,
        split=SPLIT,
        commit_hash: str = COMMIT_HASH,
        confinement_instructions: str = CONFINEMENT_INSTRUCTIONS,
        **kwargs,
    ):
        """
        Initialize PubMedMCQA dataset.

        Args:
            **kwargs: Additional arguments passed to base class
        """
        self.dataset = None
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            commit_hash=commit_hash,
            confinement_instructions=confinement_instructions,
            **kwargs,
        )

    def format_item(self, sample: Dict[str, Any], **kwargs):
        input_text = self._format_question(sample["data"])

        # Parse correct answer from Correct Option field
        correct_option = sample["data"]["Correct Option"]
        context = "\n".join(sample["data"]["Context"])
        prompt = self.confinement_instructions.replace("<CONTEXT>", context).replace(
            "<QUESTION>", input_text
        )

        processed_sample = DataLoaderIterable(
            input=prompt,
            expected_output=correct_option,
        )

        return processed_sample
