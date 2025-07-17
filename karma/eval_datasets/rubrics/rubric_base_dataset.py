"""
Health-Bench-Eval-OSS-2025-07 dataset implementation.

This module provides the HealthBenchDataset class that implements the
multimodal dataset interface for health benchmark evaluation with rubric-based scoring.
"""

import logging
from typing import Dict, Any, Tuple
from karma.data_models.dataloader_iterable import (
    DataLoaderIterable,
    ConversationTurn,
    Conversation,
    RubricCriteria,
)
from karma.eval_datasets.base_dataset import BaseMultimodalDataset

logger = logging.getLogger(__name__)


class RubricBaseDataset(BaseMultimodalDataset):
    """
    Health-Bench-Eval-OSS-2025-07
    Handles medical question answering with rubric-based evaluation.
    We are considering the first ideal completion to evaluate
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        system_prompt: str,
        **kwargs,
    ):
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            **kwargs,
        )
        self.system_prompt = system_prompt

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        # Extract prompt information
        conversation = []
        for conversation_turn in sample["prompt"]:
            conversation.append(
                ConversationTurn(
                    content=conversation_turn["content"],
                    role=conversation_turn["role"],
                )
            )
        conversation = Conversation(conversation_turns=conversation)

        criterions = []
        for rubric_item in sample["rubrics"]:
            criterions.append(
                RubricCriteria(
                    criterion=rubric_item["criterion"],
                    points=rubric_item["points"],
                    tags=rubric_item.get("tags", []),
                )
            )

        processed_sample = DataLoaderIterable(
            conversation=conversation,
            rubric_to_evaluate=criterions,
            system_prompt=self.system_prompt,
            other_args={"additional_info": sample.get("ideal_completions_data")},
        )

        return processed_sample

    def extract_prediction(self, response: str) -> Tuple[str, bool]:
        """
        Extract the prediction from model response.

        For rubric evaluation, we return the full response as the prediction.
        The actual scoring will be handled by the rubric_evaluation metric.

        Args:
            response: Model's response text

        Returns:
            Tuple of (prediction, success_flag)
        """
        # For rubric evaluation, return the full response
        return response.strip(), True
