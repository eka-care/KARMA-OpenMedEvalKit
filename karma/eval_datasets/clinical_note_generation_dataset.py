"""
Clinical Note Generation dataset implementation.

This module provides the ClinicalNoteGenerationDataset class that implements the
dataset interface for clinical note generation evaluation with rubric-based scoring.
"""

import logging
from typing import Dict, Any, Tuple
from karma.data_models.dataloader_iterable import (
    DataLoaderIterable,
)
from karma.registries.dataset_registry import register_dataset
from karma.eval_datasets.base_dataset import BaseMultimodalDataset

logger = logging.getLogger(__name__)

DATASET_NAME = "ekacare/clinical_note_generation_dataset"
SPLIT = "test"  # Adjust as needed based on actual dataset splits


@register_dataset(
    DATASET_NAME,
    split=SPLIT,
    metrics=["json_rubric_evaluation"],
    task_type="text_to_json",
)
class ClinicalNoteGenerationDataset(BaseMultimodalDataset):
    """
    Clinical Note Generation PyTorch Dataset implementing the dataset interface.
    Handles clinical note generation from conversational text with rubric-based evaluation.
    
    Dataset columns:
    - text: Conversation transcription between patient and doctor
    - sample_prompt: System prompt with clinical note generation instructions and schema
    - rubrics: Evaluation criteria for judging the generated JSON output
    
    The evaluation process:
    1. Send conversation text to the model being evaluated with sample_prompt as system prompt
    2. Model generates structured JSON/YML from conversation
    3. Send model output + rubrics to the judge LLM for evaluation
    4. Get rubric scores for batch evaluation
    """

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        **kwargs,
    ):
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            **kwargs,
        )

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        """
        Format a sample into a clinical note generation evaluation item.

        Args:
            sample: A single sample from the dataset containing:
                - text: Conversation transcription between patient and doctor
                - sample_prompt: System prompt with clinical note generation instructions
                - rubrics: Evaluation criteria for the generated JSON

        Returns:
            DataLoaderIterable containing the formatted sample for text-to-JSON evaluation
        """
        # Extract data from the sample
        conversation_text = sample.get("text", "")
        sample_prompt = sample.get("sample_prompt", "")
        rubrics = sample.get("rubrics", "")
        
        # Validate required fields
        if not conversation_text:
            logger.warning("No conversation text found in dataset sample")
            conversation_text = "No conversation provided."
            
        if not sample_prompt:
            logger.warning("No sample_prompt found in dataset sample")
            sample_prompt = "You are a medical assistant. Convert the conversation into structured format."
            
        if not rubrics:
            logger.warning("No rubrics found in dataset sample")
        
        rubric_passes = []
        if rubrics:
            rubric_passes.append({
                "name": "rubric_evaluation",
                "prompt": rubrics,
                "response_scores_map_key": "rubric_scores",
                "id_field": "rubric_id",
                "score_field": "score",
                "axes": {"*": [{"field": "score", "passes_when": {"eq": 1}}]},
            })

        other_args = {
            "document_type": "clinical_note",
            "rubric_passes": rubric_passes,
        }

        return DataLoaderIterable(
            input=conversation_text,
            system_prompt=sample_prompt,
            other_args=other_args,
        )


    def extract_prediction(self, response: str, **kwargs) -> Tuple[str, bool]:
        """
        Extract the prediction from model response.

        For clinical note generation evaluation, we return the full JSON response 
        as the prediction. The actual scoring will be handled by the 
        json_rubric_evaluation metric which will evaluate the
        JSON against the rubrics in a single batch call.

        Args:
            response: Model's JSON response text

        Returns:
            Tuple of (prediction, success_flag)
        """
        # For rubric evaluation, return the full response
        return response.strip(), True