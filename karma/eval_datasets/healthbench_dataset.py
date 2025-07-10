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
from karma.registries.dataset_registry import register_dataset
from karma.eval_datasets.base_dataset import BaseMultimodalDataset

logger = logging.getLogger(__name__)

DATASET_NAME = "Tonic/Health-Bench-Eval-OSS-2025-07"
SPLIT = "oss_eval"
COMMIT_HASH = "0865a52cdf7ed7eff9923fe0dca419d9a0d6acbf"


@register_dataset(
    DATASET_NAME,
    split=SPLIT,
    commit_hash=COMMIT_HASH,
    metrics=["healthbench_rubric_evaluation"],
    task_type="rubric_evaluation",
)
class HealthBenchDataset(BaseMultimodalDataset):
    """
    Health-Bench-Eval-OSS-2025-07 PyTorch Dataset implementing the multimodal interface.
    Handles medical question answering with rubric-based evaluation.
    We are considering the first ideal completion to evaluate
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
        # Extract prompt information
        conversation = []
        for conversation_turn in sample["prompt"]:
            conversation.append(
                ConversationTurn(
                    content=conversation_turn["content"],
                    role=conversation_turn["role"],
                )
            )
        conversation = Conversation(conversation=conversation)

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
