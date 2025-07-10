"""
Health-Bench-Eval-OSS-2025-07 dataset implementation.

This module provides the HealthBenchDataset class that implements the
multimodal dataset interface for health benchmark evaluation with rubric-based scoring.
"""

import logging
from typing import Dict, Any, Tuple

from sklearn.tree._criterion import Criterion

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

HEALTHBENCH_PROMPT = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
{conversation}
# Rubric item
{rubric}

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the criteria says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()
# the above prompt has been taken verbatim from the healthbench eval repository


@register_dataset(
    DATASET_NAME,
    split=SPLIT,
    commit_hash=COMMIT_HASH,
    metrics=["accuracy"],  # Placeholder metric until rubric_evaluation is implemented
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
        for i in sample["rubrics"]:
            criterions.append(RubricCriteria(criterion=i))

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
