"""
Medical Records Parsing dataset implementation.

This module provides the MedicalRecordsParsingDataset class that implements the
dataset interface for medical records parsing evaluation with rubric-based scoring.
"""

import logging
from typing import Dict, Any, Tuple
from datasets import Image
from karma.data_models.dataloader_iterable import (
    DataLoaderIterable,
)
from karma.registries.dataset_registry import register_dataset
from karma.eval_datasets.base_dataset import BaseMultimodalDataset

logger = logging.getLogger(__name__)

DATASET_NAME = "ekacare/medical_records_parsing_validation_set"
SPLIT = "test"  # Adjust as needed based on actual dataset splits


@register_dataset(
    DATASET_NAME,
    split=SPLIT,
    metrics=["json_rubric_evaluation"],
    task_type="image_to_json",
    optional_args=["system_prompt"],
)
class MedicalRecordsParsingDataset(BaseMultimodalDataset):
    """
    Medical Records Parsing PyTorch Dataset implementing the dataset interface.
    Handles medical records parsing with rubric-based evaluation.
    
    Dataset columns:
    - image: Medical record image (Lab-report or Prescription)
    - sample_prompt: The parsing prompt to send to the model
    - rubrics: Evaluation criteria for judging the model response
    - document_type: Type of document (Lab-report or Prescription)
    
    The evaluation process:
    1. Send sample_prompt + image to the model being evaluated
    2. Send model response + rubrics to the judge LLM for evaluation
    3. Get rubric scores for batch evaluation
    """

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        system_prompt: str = "You are an expert at parsing medical records. Extract the information accurately from the provided medical record image.",
        **kwargs,
    ):
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            **kwargs,
        )
        self.system_prompt = system_prompt
        # Cast the image column to handle image bytes properly
        self.dataset = self.dataset.cast_column("image", Image(decode=False))

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        """
        Format a sample into a medical records parsing evaluation item.

        Args:
            sample: A single sample from the dataset containing:
                - image: Medical record image
                - sample_prompt: Parsing prompt for the model
                - rubrics: Evaluation criteria 
                - document_type: Type of document (Lab-report or Prescription)

        Returns:
            DataLoaderIterable containing the formatted sample for multimodal evaluation
        """
        # Extract data from the sample
        image = sample["image"]["bytes"]
        sample_prompt = sample.get("sample_prompt", "")
        rubrics = sample.get("rubrics", "")
        document_type = sample.get("document_type", "unknown")
        
        # Validate required fields
        if not sample_prompt:
            logger.warning("No sample_prompt found in dataset sample")
            sample_prompt = "Parse the medical record image and extract all relevant information."
            
        if not rubrics:
            logger.warning("No rubrics found in dataset sample")
        
        # Store document type and rubrics for the metric to access
        other_args = {
            "rubric_prompt": rubrics,
            "document_type": document_type,
        }
        
        # Create the DataLoaderIterable for the model evaluation
        processed_sample = DataLoaderIterable(
            input=sample_prompt,
            images=[image],  # Include image for multimodal models
            system_prompt=self.system_prompt,
            other_args=other_args,  # Store rubric and metadata for metric access
        )

        return processed_sample
