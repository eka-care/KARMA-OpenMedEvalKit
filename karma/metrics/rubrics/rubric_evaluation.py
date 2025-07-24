import json
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import List, Dict, Any, Optional
from karma.metrics.base_metric_abs import BaseMetric
from karma.models.aws_bedrock import AWSBedrock
from karma.models.openai_llm import OpenAILLM
from karma.registries.metrics_registry import register_metric
from karma.data_models.dataloader_iterable import (
    DataLoaderIterable,
    RubricCriteria,
    Conversation,
    ConversationTurn,
)

logger = logging.getLogger(__name__)

PROMPT = """
Your job is to look at a Input and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Input
<<input_prompt>>
# Rubric item
<<rubric>>

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
Return just the json, no need for markdown beginning. Do not include any other text in the response.
""".strip()
# the above prompt has been taken verbatim from the healthbench eval repository

BATCH_PROMPT = """
Your job is to look at a Input and multiple rubric items, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows each rubric item.

# Input
<<input_prompt>>

# Rubric items
<<rubrics>>

# Instructions
Return a JSON array where each element corresponds to a rubric item (in the same order as provided) and contains the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example
If there are 2 rubric items, your response should be a JSON array with 2 elements:
```json
[
  {
    "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
    "criteria_met": false
  },
  {
    "explanation": "The assistant provided accurate information about the topic.",
    "criteria_met": true
  }
]
```

# Final instruction
Return just the JSON array, no need for markdown beginning. Do not include any other text in the response.
""".strip()


@register_metric(
    name="rubric_evaluation",
    optional_args=["batch_size", "max_workers"],
    default_args={"batch_size": 100, "max_workers": 4},
)
class RubricMetric(BaseMetric):
    """
    LLM driven rubric evaluation metric.
    """

    def __init__(
        self,
        metric_name,
        provider_to_use,
        model_id,
        batch_size=1,
        max_workers=4,
        **kwargs,
    ):
        super().__init__(metric_name=metric_name, **kwargs)
        self.provider = provider_to_use
        if isinstance(batch_size, str):
            batch_size = int(batch_size)
        self.batch_size = batch_size
        if isinstance(max_workers, str):
            max_workers = int(max_workers)
        self.max_workers = max_workers
        logger.info(
            f"Got {provider_to_use} rubric evaluation metric with batch_size={batch_size}, max_workers={max_workers}"
        )
        if self.provider == "openai":
            self.model = OpenAILLM(model_name_or_path=model_id)
        elif self.provider == "bedrock":
            self.model = AWSBedrock(model_name_or_path=model_id)

    def evaluate(self, predictions, references=None, rubrics=None, **kwargs):
        """
        Evaluate predictions against rubrics using LLM-based scoring.

        Args:
            predictions: List of conversation objects (DataLoaderIterable)
            references: Not used in rubric evaluation
            rubrics: Not used - rubrics are embedded in predictions
            **kwargs: Additional arguments

        Returns:
            Dict containing evaluation results
        """
        samples = kwargs["samples"]
        logger.info(
            f"Evaluating {len(predictions)} conversations with {self.provider} model - {self.model}"
        )

        # Handle empty predictions
        if not predictions:
            return {"rubric_evaluation": self._aggregate_results([])}

        # Use ThreadPoolExecutor for parallel conversation processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all conversation processing tasks and track their order
            future_to_index = {
                executor.submit(
                    self._process_single_conversation,
                    prediction,
                    sample,
                    sample_rubrics,
                ): i
                for i, (prediction, sample, sample_rubrics) in enumerate(
                    zip(predictions, samples, rubrics)
                )
            }

            # Initialize results list with correct size
            question_results = [None] * len(predictions)

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                result = future.result()
                question_results[index] = result

        # Aggregate results
        return {"rubric_evaluation": self._aggregate_results(question_results)}

    def _evaluate_batch(
        self, conversation_json: str, rubric_batch: List[RubricCriteria]
    ) -> List[Dict]:
        """
        Evaluate a batch of rubrics for a single conversation.

        Args:
            conversation_json: JSON representation of the conversation
            rubric_batch: List of rubric criteria to evaluate

        Returns:
            List of evaluation results, one per rubric
        """
        if len(rubric_batch) == 1:
            # Use single rubric prompt for batch size 1
            prompt = PROMPT.replace("<<input_prompt>>", conversation_json).replace(
                "<<rubric>>", rubric_batch[0].model_dump_json()
            )

            eval_input = DataLoaderIterable(
                input=prompt,
                system_prompt="You are an expert evaluator for medical question answering.",
            )

            response = self.model.run([eval_input])[0]

            try:
                eval_result = json.loads(response)
                return [
                    {
                        "criteria_met": eval_result["criteria_met"],
                        "explanation": eval_result["explanation"],
                        "rubric": rubric_batch[0],
                    }
                ]
            except json.JSONDecodeError:
                return [
                    {
                        "criteria_met": False,
                        "explanation": f"Failed to parse response: {response}",
                        "rubric": rubric_batch[0],
                    }
                ]
        else:
            # Use batch prompt for multiple rubrics
            rubrics_json = json.dumps([rubric.model_dump() for rubric in rubric_batch])
            prompt = BATCH_PROMPT.replace(
                "<<input_prompt>>", conversation_json
            ).replace("<<rubrics>>", rubrics_json)

            eval_input = DataLoaderIterable(
                input=prompt,
                system_prompt="You are an expert evaluator for medical question answering.",
            )

            response = self.model.run([eval_input])[0]

            try:
                eval_results = json.loads(response)
                if not isinstance(eval_results, list) or len(eval_results) != len(
                    rubric_batch
                ):
                    raise ValueError(
                        f"Expected {len(rubric_batch)} results, got {len(eval_results) if isinstance(eval_results, list) else 'non-list'}"
                    )
                logger.info(f"Eval results: {eval_results} for {conversation_json}")
                return [
                    {
                        "criteria_met": result["criteria_met"],
                        "explanation": result["explanation"],
                        "rubric": rubric,
                    }
                    for result, rubric in zip(eval_results, rubric_batch)
                ]
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # Fallback to individual evaluation if batch fails
                logger.warning(
                    f"Batch evaluation failed: {e}. Falling back to individual evaluation."
                )
                return self._evaluate_individual_fallback(
                    conversation_json, rubric_batch
                )

    def _evaluate_individual_fallback(
        self, conversation_json: str, rubric_batch: List[RubricCriteria]
    ) -> List[Dict]:
        """
        Fallback to individual evaluation when batch processing fails.

        Args:
            conversation_json: JSON representation of the conversation
            rubric_batch: List of rubric criteria to evaluate

        Returns:
            List of evaluation results, one per rubric
        """
        results = []
        for rubric in rubric_batch:
            prompt = PROMPT.replace("<<input_prompt>>", conversation_json).replace(
                "<<rubric>>", rubric.model_dump_json()
            )

            eval_input = DataLoaderIterable(
                input=prompt,
                system_prompt="You are an expert evaluator for medical question answering.",
            )

            response = self.model.run([eval_input])[0]

            try:
                eval_result = json.loads(response)
                results.append(
                    {
                        "criteria_met": eval_result["criteria_met"],
                        "explanation": eval_result["explanation"],
                        "rubric": rubric,
                    }
                )
            except json.JSONDecodeError:
                results.append(
                    {
                        "criteria_met": False,
                        "explanation": f"Failed to parse response: {response}",
                        "rubric": rubric,
                    }
                )

        return results

    def _process_single_conversation(self, prediction, sample, sample_rubrics):
        """
        Process a single conversation and its rubrics.

        Args:
            prediction: The prediction text for this conversation
            sample: The sample containing conversation data
            sample_rubrics: List of rubric criteria for this sample

        Returns:
            Dict containing rubric evaluations and question score
        """
        # Format conversation as string
        # sample.conversation.conversation_turns.append(
        #     ConversationTurn(content=prediction, role="assistant")
        # )
        sample.conversation.conversation_turns = [
            ConversationTurn(content=prediction, role="assistant")
        ]

        # Get conversation JSON once
        conversation_json = sample.conversation.model_dump_json()

        # Evaluate rubrics in batches
        grading_responses = []
        for i in range(0, len(sample_rubrics), self.batch_size):
            batch_end = min(i + self.batch_size, len(sample_rubrics))
            rubric_batch = sample_rubrics[i:batch_end]

            # Evaluate this batch
            batch_results = self._evaluate_batch(conversation_json, rubric_batch)
            grading_responses.extend(batch_results)

        # Calculate score for this question
        question_score = self.calculate_score(sample_rubrics, grading_responses)

        return {
            "rubric_evaluations": grading_responses,
            "question_score": question_score,
        }

    def calculate_score(
        self, rubric_items: List[RubricCriteria], grading_responses: List[Dict]
    ) -> Optional[float]:
        """
        Calculate the score for a single question based on rubric evaluations.

        Args:
            rubric_items: List of RubricCriteria objects
            grading_responses: List of grading responses from the model

        Returns:
            Score as a float between 0 and 1, or None if no positive point criteria
        """
        # Calculate total possible points (only positive point criteria)
        total_possible_points = sum(
            rubric.points for rubric in rubric_items if rubric.points > 0
        )

        # Return None if no positive point criteria exist
        if total_possible_points == 0:
            return None

        # Calculate achieved points
        achieved_points = sum(
            rubric.points
            for rubric, grading_response in zip(rubric_items, grading_responses)
            if grading_response["criteria_met"]
        )

        # Calculate overall score as ratio
        overall_score = achieved_points / total_possible_points
        return overall_score

    def _aggregate_results(self, question_results: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate results across all questions.

        Args:
            question_results: List of question-level results

        Returns:
            Dict containing aggregated metrics
        """
        # Filter out questions with None scores
        valid_scores = [
            result["question_score"]
            for result in question_results
            if result["question_score"] is not None
        ]

        if not valid_scores:
            return {
                "overall_score": 0.0,
                "num_questions": len(question_results),
                "num_valid_questions": 0,
                "question_results": question_results,
            }

        # Calculate overall metrics
        overall_score = np.mean(valid_scores)
        std_dev = np.std(valid_scores, ddof=1) if len(valid_scores) > 1 else 0.0

        # Calculate bootstrap standard error (simplified)
        bootstrap_std = (
            std_dev / np.sqrt(len(valid_scores)) if len(valid_scores) > 0 else 0.0
        )

        # Aggregate by tags if available
        tag_scores = self._aggregate_by_tags(question_results)

        return {
            "overall_score": float(overall_score),
            "std_dev": float(std_dev),
            "bootstrap_std": float(bootstrap_std),
            "num_questions": len(question_results),
            "num_valid_questions": len(valid_scores),
            "tag_scores": tag_scores,
            # "question_results": question_results
        }

    def _aggregate_by_tags(
        self, question_results: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate scores by rubric tags.

        Args:
            question_results: List of question-level results

        Returns:
            Dict of tag -> aggregated metrics
        """
        tag_scores = {}

        for result in question_results:
            if result["question_score"] is None:
                continue

            for evaluation in result["rubric_evaluations"]:
                rubric = evaluation["rubric"]
                for tag in rubric.tags:
                    if tag not in tag_scores:
                        tag_scores[tag] = []

                    # For tag-level scoring, we consider individual rubric performance
                    if evaluation["criteria_met"]:
                        tag_scores[tag].append(1.0)
                    else:
                        tag_scores[tag].append(0.0)

        # Aggregate tag scores
        aggregated_tags = {}
        for tag, scores in tag_scores.items():
            if scores:
                aggregated_tags[tag] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                    "count": len(scores),
                }

        return aggregated_tags
