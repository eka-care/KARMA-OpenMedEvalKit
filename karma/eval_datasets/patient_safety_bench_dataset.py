"""
PatientSafetyBench dataset implementation for patient safety evaluation.

This module provides the PatientSafetyBenchDataset class that implements the multimodal
dataset interface for evaluating the safety of medical language models from the patient's
perspective.

PatientSafetyBench: A benchmark of patient-oriented queries that test critical safety
policies, enabling measurement of how well medical LLMs avoid harmful, misleading,
unlicensed, or discriminatory responses.
"""

import logging
from typing import Any, Dict, Tuple

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset

logger = logging.getLogger(__name__)

# Dataset constants
DATASET_NAME = "microsoft/PatientSafetyBench"
SPLIT = "train"
COMMIT_HASH = "c39259fb04de6d4615cd3ee0f5d30497872b5226"  # Use latest version

# Policy categories - we exclude category 2 (Misdiagnosis and Overconfidence)
POLICY_CATEGORIES = {
    1: "Harmful or Dangerous Medical Advice",
    3: "Unlicensed Practice of Medicine",
    4: "Health Misinformation",
    5: "Bias, Discrimination, and Stigmatization",
}


@register_dataset(
    DATASET_NAME,
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    metrics=["accuracy"],
    task_type="safety_classification",
)
class PatientSafetyBenchDataset(BaseMultimodalDataset):
    """
    PatientSafetyBench PyTorch Dataset for patient-oriented safety evaluation.
    
    This dataset filters samples to exclude category 2 and uses the content field
    to evaluate how well medical LLMs avoid harmful, misleading, or discriminatory
    responses when interacting with patients.
    """

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        commit_hash: str = COMMIT_HASH,
        **kwargs,
    ):
        """
        Initialize PatientSafetyBench dataset.

        Args:
            dataset_name: Name of the dataset (default: microsoft/PatientSafetyBench)
            split: Dataset split to use (default: train)
            commit_hash: Commit hash of the dataset version
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            commit_hash=commit_hash,
            **kwargs,
        )

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        """
        Format a sample into a text prompt for safety evaluation.

        For PatientSafetyBench, we filter on category (exclude category 2)
        and use the content field directly for evaluation.

        Args:
            sample: A single sample from the dataset

        Returns:
            DataLoaderIterable with the content prompt or None if filtered out
        """
        # Filter out category 2 (Misdiagnosis and Overconfidence)
        category = sample.get("category", None)
        if category == 2:
            return None

        content = sample.get("content", "")
        
        processed_sample = DataLoaderIterable(
            input=content,
            expected_output=None,  # Safety evaluation - no specific expected output
            other_args={
                "category": category,
                "policy": POLICY_CATEGORIES.get(category, "Unknown"),
            },
        )
        
        return processed_sample

    def extract_prediction(self, response: str, **kwargs) -> Tuple[str, bool]:
        """
        Extract the model's response.

        For PatientSafetyBench, we return the response as-is since safety scoring
        is typically done through separate evaluation (e.g., using an LLM-as-judge).

        Args:
            response: The model's response text

        Returns:
            Tuple of (response, success_flag)
        """
        extracted_response = response.strip()
        
        if not extracted_response:
            logger.warning("Empty response received")
            return "", False
        
        return extracted_response, True

    def __iter__(self):
        """
        Iterate through the dataset, filtering samples.

        Yields:
            Formatted samples that are not in category 2
        """
        sample_count = 0
        for sample in self.dataset:
            formatted = self.format_item(sample)
            if formatted is not None:  # Skip filtered out samples
                yield formatted
                sample_count += 1
                if sample_count >= self.max_samples:
                    break
