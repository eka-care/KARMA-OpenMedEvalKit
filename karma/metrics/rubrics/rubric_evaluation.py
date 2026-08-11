import json
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import List, Dict, Any, NamedTuple, Optional, Tuple
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


class _EvalSpec(NamedTuple):
    """One assembled judge call for a rubric batch — built once, hashed and run."""

    eval_input: DataLoaderIterable
    rubric_batch: List[RubricCriteria]
    use_structured: bool
    is_single: bool

PROMPT = """
Your job is to evaluate how well the last assistant response (i.e., the completion) satisfies the rubric item below, based on its intended clinical meaning rather than exact phrasing.

# Input
<<input_prompt>>

# Rubric item
<<rubric>>

# Instructions
Return a JSON object with two fields: "explanation" and "criteria_met".

### Evaluation principles
- Assess whether the **intended medical or conceptual meaning** of the rubric item is clearly expressed in the response, not whether the exact words match.
- Accept **paraphrases, synonyms, or equivalent Indian clinical terms** if they convey the same meaning accurately.
- A broader or logically encompassing statement (e.g., "RAAS blockade" instead of "ACE inhibitors") counts as correct if the clinical intent matches.
- Mark **criteria_met = true** only if the central idea or fact in the rubric is fully satisfied and factually correct.
- Mark **criteria_met = false** if the response omits, contradicts, or misstates that idea.
- Do not penalize for order, tone, or stylistic differences if the correct clinical meaning is conveyed.
- Ignore superficial mismatches (e.g., punctuation, ordering, format) that do not change meaning.
- When a rubric item contains examples ("such as", "for example", "including"), the response may include any equivalent or representative example to meet the criterion.
- Always prioritize **factual correctness and conceptual completeness** over literal phrasing.

### Example 1
If the conversation is
"user: Should I go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better"
and the rubric item is
"tells the user to go to the hospital if unconscious",
you should return:

```json
{
  "explanation": "The assistant does not instruct the user to go to the hospital when unconscious, so the clinical intent of the rubric is not satisfied.",
  "criteria_met": false
}
""".strip()
# the above prompt has been taken verbatim from the healthbench eval repository

PROMPT_MULTI_TURN = """
Your job is to evaluate how well the conversation satisfies the rubric item below. For single-turn conversations, evaluate the assistant's response. For multi-turn conversations, evaluate the entire conversation flow.

# Conversation
<<input_prompt>>

# Rubric item
<<rubric>>

# Instructions
Return a JSON object with two fields: "explanation" and "criteria_met".

### Evaluation principles
- Assess whether the **intended medical or conceptual meaning** of the rubric item is clearly expressed in the conversation, not whether the exact words match.
- For multi-turn conversations, consider the **entire conversation flow**, not just individual responses.
- Accept **paraphrases, synonyms, or equivalent clinical terms** if they convey the same meaning accurately.
- A broader or logically encompassing statement (e.g., "RAAS blockade" instead of "ACE inhibitors") counts as correct if the clinical intent matches.
- Mark **criteria_met = true** only if the central idea or fact in the rubric is fully satisfied and factually correct.
- Mark **criteria_met = false** if the conversation omits, contradicts, or misstates that idea.
- Do not penalize for order, tone, or stylistic differences if the correct clinical meaning is conveyed.
- Ignore superficial mismatches (e.g., punctuation, ordering, format) that do not change meaning.
- When a rubric item contains examples ("such as", "for example", "including"), the conversation may include any equivalent or representative example to meet the criterion.
- Always prioritize **factual correctness and conceptual completeness** over literal phrasing.

### Example 1
If the conversation is
"user: Should I go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better"
and the rubric item is
"tells the user to go to the hospital if unconscious",
you should return:

```json
{
  "explanation": "The assistant does not instruct the user to go to the hospital when unconscious, so the clinical intent of the rubric is not satisfied.",
  "criteria_met": false
}
""".strip()

BATCH_PROMPT = """
Your job is to evaluate how well the last assistant response (i.e., the completion) satisfies each rubric item below, based on the intended *clinical meaning* rather than exact wording.

# Input
<<input_prompt>>

# Rubric items
<<rubrics>>

# Instructions
Return a JSON array where each element corresponds to a rubric item (in the same order as provided) and contains two fields: "explanation" and "criteria_met".

### Evaluation principles
- Evaluate whether the **intended clinical concept or takeaway** in each rubric item is clearly conveyed in the response, regardless of phrasing or order.
- Do **not** require exact words, synonyms are acceptable if the **same medical meaning** is expressed.
- If the response shows an *equivalent reasoning chain* or *clinically correct paraphrase*, count it as meeting the rubric.
- A rubric should only be marked false when its central idea or clinical fact is **missing, incorrect, or contradicted**.
- Minor differences in adjectives, examples, or emphasis should not affect scoring.
- When examples are mentioned ("such as", "for example", "including"), the response may include any equivalent or representative example and still meet the criterion.
- If the response mentions a broader or logically encompassing statement that covers the rubric's intent (e.g., "RAAS blockers" instead of "ACE inhibitors"), mark as met.
- Do not reward vague mentions that lack clinical specificity or factual correctness; correctness takes precedence over surface similarity.

### Scoring logic
- For each rubric item:
  - Provide an "explanation" briefly summarizing how the response aligns or fails to align with the rubric's medical meaning.
  - Set "criteria_met" to true only if the **core clinical idea** is fully satisfied.
  - Set "criteria_met" to false if the response omits, misstates, or contradicts the central takeaway.

### Example output
If there are 2 rubric items, your output should look like this:
[
  {
    "explanation": "The assistant correctly states that ACE inhibitors are first-line for CKD, satisfying the intended clinical meaning.",
    "criteria_met": true
  },
  {
    "explanation": "The assistant mentions hypertension but omits renoprotective effects, so the core takeaway is missing.",
    "criteria_met": false
  }
]

# Final instruction
Return just the JSON array, with no markdown or extra commentary.
""".strip()

BATCH_PROMPT_MULTI_TURN = """
Your job is to evaluate how well the conversation satisfies each rubric item below. For single-turn conversations, evaluate the assistant's response. For multi-turn conversations, evaluate the entire conversation flow.

# Conversation
<<input_prompt>>

# Rubric items
<<rubrics>>

# Instructions
Return a JSON array where each element corresponds to a rubric item (in the same order as provided) and contains two fields: "explanation" and "criteria_met".

### Evaluation principles
- Evaluate whether the **intended clinical concept or takeaway** in each rubric item is clearly conveyed in the conversation, regardless of phrasing or order.
- For multi-turn conversations, consider the **entire conversation flow**, not just individual responses.
- Do **not** require exact words, synonyms are acceptable if the **same medical meaning** is expressed.
- If the conversation shows an *equivalent reasoning chain* or *clinically correct paraphrase*, count it as meeting the rubric.
- A rubric should only be marked false when its central idea or clinical fact is **missing, incorrect, or contradicted**.
- Minor differences in adjectives, examples, or emphasis should not affect scoring.
- When examples are mentioned ("such as", "for example", "including"), the conversation may include any equivalent or representative example and still meet the criterion.
- If the conversation mentions a broader or logically encompassing statement that covers the rubric's intent (e.g., "RAAS blockers" instead of "ACE inhibitors"), mark as met.
- Do not reward vague mentions that lack clinical specificity or factual correctness; correctness takes precedence over surface similarity.

### Scoring logic
- For each rubric item:
  - Provide an "explanation" briefly summarizing how the conversation aligns or fails to align with the rubric's medical meaning.
  - Set "criteria_met" to true only if the **core clinical idea** is fully satisfied.
  - Set "criteria_met" to false if the conversation omits, misstates, or contradicts the central takeaway.

### Example output
If there are 2 rubric items, your output should look like this:
[
  {
    "explanation": "The assistant correctly states that ACE inhibitors are first-line for CKD, satisfying the intended clinical meaning.",
    "criteria_met": true
  },
  {
    "explanation": "The assistant mentions hypertension but omits renoprotective effects, so the core takeaway is missing.",
    "criteria_met": false
  }
]

# Final instruction
Return just the JSON array, with no markdown or extra commentary.
""".strip()

# Structured template for datasets that provide their own instructions/definitions/guidelines
BATCH_PROMPT_STRUCTURED = """
<<instructions>>

# Conversation
<<input_prompt>>

# Rubric Definitions
<<rubric_definitions>>

# Rubric Criteria
<<rubrics>>

# Evaluation Guidelines
<<output_guidelines>>
""".strip()


@register_metric(
    name="rubric_evaluation",
    optional_args=[
        "batch_size",
        "max_workers",
        "use_multi_turn_prompts",
        "use_serialized_conversations",
    ],
    default_args={
        "batch_size": 100,
        "max_workers": 20,
        "use_multi_turn_prompts": False,
        "use_serialized_conversations": False,
    },
)
class RubricMetric(BaseMetric):
    """
    LLM driven rubric evaluation metric.
    """

    non_percentage_fields = frozenset({"num_questions", "num_valid_questions"})

    def __init__(
        self,
        metric_name,
        provider_to_use,
        model_id,
        batch_size=100,
        max_workers=4,
        use_multi_turn_prompts=False,
        use_serialized_conversations=False,
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
        if isinstance(use_multi_turn_prompts, str):
            use_multi_turn_prompts = use_multi_turn_prompts.lower() == "true"
        self.use_multi_turn_prompts = use_multi_turn_prompts
        if isinstance(use_serialized_conversations, str):
            use_serialized_conversations = (
                use_serialized_conversations.lower() == "true"
            )
        self.use_serialized_conversations = use_serialized_conversations
        logger.info(
            f"Got {provider_to_use} rubric evaluation metric with batch_size={batch_size}, max_workers={max_workers}, use_multi_turn_prompts={use_multi_turn_prompts}, use_serialized_conversations={use_serialized_conversations}"
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
        serialized_samples = kwargs.get("serialized_samples")
        cache_manager = kwargs.get("cache_manager")
        dataset_name = kwargs.get("dataset_name")
        logger.info(
            f"Evaluating {len(predictions)} conversations with {self.provider} model - {self.model}"
        )

        # Handle empty predictions
        if not predictions:
            aggregated_summary, per_sample_details = self._aggregate_results([])
            return {
                "rubric_evaluation": aggregated_summary,
                "rubric_evaluation_details": per_sample_details,
            }

        # Prepare each conversation (append the model output / parse pass-through)
        # up front so judge_hash lookups can be batched in one query. The hash is
        # taken over the assembled eval_input(s) the judge actually receives, so
        # any prompt/rubric/output change invalidates the cache automatically.
        judge_model_config = self._build_judge_model_config()
        prepared = []
        judge_hashes = []
        for prediction, sample, sample_rubrics in zip(predictions, samples, rubrics):
            conv = self._prepare_conversation(prediction, sample)
            judge_input = self._build_judge_input(conv, sample_rubrics)
            # Build the judge inputs once; they are hashed for the cache key here
            # and run directly on a miss (no second assembly).
            specs = self._build_eval_specs(conv, sample_rubrics)
            prepared.append((conv, judge_input, specs))
            if cache_manager:
                judge_hashes.append(
                    cache_manager.generate_hash(
                        {
                            "eval_inputs": [spec.eval_input.model_dump() for spec in specs],
                            "judge_model_config": judge_model_config,
                        }
                    )
                )
            else:
                judge_hashes.append(None)

        cached_by_hash = {}
        if cache_manager:
            wanted = [h for h in judge_hashes if h]
            if wanted:
                try:
                    cached_by_hash = cache_manager.get_rubric_outputs(wanted)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning("Rubric cache lookup failed: %s", exc)

        hits = sum(1 for h in judge_hashes if h and h in cached_by_hash)
        if cache_manager:
            logger.info(
                f"[rubric_evaluation] judge cache: {hits}/{len(predictions)} reused"
            )

        # Use ThreadPoolExecutor for parallel conversation processing (misses only
        # call the judge; hits reconstruct from the cached row).
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(
                    self._score_conversation,
                    prepared[i],
                    rubrics[i],
                    cached_by_hash.get(judge_hashes[i]) if judge_hashes[i] else None,
                ): i
                for i in range(len(predictions))
            }

            # Initialize results list with correct size
            question_results = [None] * len(predictions)

            # Collect results as they complete
            completed = 0
            total = len(predictions)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                result = future.result()
                question_results[index] = result
                completed += 1
                logger.info(f"[rubric_evaluation] {completed}/{total} conversations evaluated")

        # Attach the eval_input-derived judge_hash so persistence keys on the
        # exact value looked up above (no recompute drift).
        for i, result in enumerate(question_results):
            if result is not None:
                result["judge_hash"] = judge_hashes[i]

        # Aggregate results
        aggregated_summary, per_sample_details = self._aggregate_results(
            question_results
        )

        if cache_manager and serialized_samples and dataset_name:
            try:
                self._persist_rubric_results(
                    cache_manager=cache_manager,
                    dataset_name=dataset_name,
                    question_results=question_results,
                    serialized_samples=serialized_samples,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to persist rubric evaluation results: %s", exc)

        if cache_manager and serialized_samples:
            try:
                self._persist_rubric_outputs(
                    cache_manager=cache_manager,
                    question_results=question_results,
                    serialized_samples=serialized_samples,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to persist rubric outputs: %s", exc)

        return {
            "rubric_evaluation": aggregated_summary,
            "rubric_evaluation_details": per_sample_details,
        }

    _JUDGE_SYSTEM_PROMPT = "You are an expert evaluator for medical question answering."

    def _build_eval_input(
        self,
        conversation_json: str,
        rubric_batch: List[RubricCriteria],
        rubric_instructions: Optional[str] = None,
        rubric_definitions: Optional[str] = None,
        rubric_output_guidelines: Optional[str] = None,
    ) -> Tuple[DataLoaderIterable, bool, bool]:
        """Assemble the exact judge input for a rubric batch.

        Returns ``(eval_input, use_structured_prompt, is_single)``. This is what
        the judge receives *and* what judge_hash is computed over, so template,
        conversation, rubrics and system prompt are all captured by one object.
        """
        use_structured_prompt = all(
            [rubric_instructions, rubric_definitions, rubric_output_guidelines]
        )
        is_single = len(rubric_batch) == 1 and not use_structured_prompt

        if is_single:
            prompt_template = (
                PROMPT_MULTI_TURN if self.use_multi_turn_prompts else PROMPT
            )
            prompt = prompt_template.replace(
                "<<input_prompt>>", conversation_json
            ).replace("<<rubric>>", rubric_batch[0].model_dump_json())
        else:
            rubrics_json = json.dumps(
                [rubric.model_dump() for rubric in rubric_batch]
            )
            if use_structured_prompt:
                prompt = (
                    BATCH_PROMPT_STRUCTURED.replace(
                        "<<instructions>>", rubric_instructions
                    )
                    .replace("<<input_prompt>>", conversation_json)
                    .replace("<<rubric_definitions>>", rubric_definitions)
                    .replace("<<rubrics>>", rubrics_json)
                    .replace("<<output_guidelines>>", rubric_output_guidelines)
                )
            else:
                prompt_template = (
                    BATCH_PROMPT_MULTI_TURN
                    if self.use_multi_turn_prompts
                    else BATCH_PROMPT
                )
                prompt = prompt_template.replace(
                    "<<input_prompt>>", conversation_json
                ).replace("<<rubrics>>", rubrics_json)

        eval_input = DataLoaderIterable(
            input=prompt, system_prompt=self._JUDGE_SYSTEM_PROMPT
        )
        return eval_input, use_structured_prompt, is_single

    def _run_eval_input(
        self,
        eval_input: DataLoaderIterable,
        rubric_batch: List[RubricCriteria],
        use_structured_prompt: bool,
        is_single: bool,
        conversation_json: str,
    ) -> Tuple[List[Dict], str]:
        """Call the judge on a prebuilt eval_input and parse its response.

        Returns ``(evaluation results, raw judge response)`` — raw is kept for
        the rubric_outputs cache, including when parsing fails.
        """
        response = self.model.run([eval_input])[0]

        if is_single:
            try:
                eval_result = json.loads(response)
                if not isinstance(eval_result, dict):
                    raise ValueError(
                        f"Expected JSON object with rubric fields, received {type(eval_result).__name__}"
                    )
                return (
                    [
                        {
                            "criteria_met": eval_result["criteria_met"],
                            "explanation": eval_result["explanation"],
                            "rubric": rubric_batch[0],
                        }
                    ],
                    response,
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                return (
                    [
                        {
                            "criteria_met": False,
                            "explanation": f"Failed to parse response: {response}",
                            "rubric": rubric_batch[0],
                        }
                    ],
                    response,
                )

        try:
            eval_results = self._parse_structured_response(
                response, rubric_batch, use_structured_prompt
            )
            return eval_results, response
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            logger.warning(
                f"Batch evaluation failed: {e}. Falling back to individual evaluation."
            )
            return (
                self._evaluate_individual_fallback(conversation_json, rubric_batch),
                response,
            )

    def _evaluate_batch(
        self,
        conversation_json: str,
        rubric_batch: List[RubricCriteria],
        rubric_instructions: Optional[str] = None,
        rubric_definitions: Optional[str] = None,
        rubric_output_guidelines: Optional[str] = None,
    ) -> Tuple[List[Dict], str]:
        """Build the judge input for a rubric batch and run it."""
        eval_input, use_structured_prompt, is_single = self._build_eval_input(
            conversation_json,
            rubric_batch,
            rubric_instructions,
            rubric_definitions,
            rubric_output_guidelines,
        )
        return self._run_eval_input(
            eval_input,
            rubric_batch,
            use_structured_prompt,
            is_single,
            conversation_json,
        )

    def _parse_structured_response(
        self,
        response: str,
        rubric_batch: List[RubricCriteria],
        use_structured_prompt: bool,
    ) -> List[Dict]:
        """
        Parse the LLM response based on the prompt format used.

        Args:
            response: Raw LLM response
            rubric_batch: List of rubric criteria that were evaluated
            use_structured_prompt: Whether structured prompt format was used

        Returns:
            List of normalized evaluation results
        """
        eval_results = json.loads(response)

        if (
            use_structured_prompt
            and isinstance(eval_results, dict)
            and "rubric_scores" in eval_results
        ):
            # Parse structured response format: {"rubric_scores": {"1": {"score": 0, "explanation": "..."}, ...}}
            rubric_scores = eval_results["rubric_scores"]
            normalized_results = []

            for rubric in rubric_batch:
                rubric_id = rubric.rubric_id or str(len(normalized_results) + 1)
                if rubric_id in rubric_scores:
                    score_data = rubric_scores[rubric_id]
                    raw_score = score_data.get("score")
                    # null score from judge (N/A) → criteria_met=None; preserves
                    # N/A distinction in the dashboard vs a genuine FAIL (0).
                    criteria_met = None if raw_score is None else (raw_score == 1)
                    normalized_results.append({
                        "criteria_met": criteria_met,
                        "explanation": score_data.get("explanation", ""),
                        "rubric": rubric,
                    })
                else:
                    logger.warning(f"Missing rubric_id {rubric_id} in response")
                    normalized_results.append(
                        {
                            "criteria_met": False,
                            "explanation": f"Rubric ID {rubric_id} not found in response",
                            "rubric": rubric,
                        }
                    )

            return normalized_results
        else:
            # Parse default array format: [{"criteria_met": true, "explanation": "..."}, ...]
            if not isinstance(eval_results, list) or len(eval_results) != len(
                rubric_batch
            ):
                raise ValueError(
                    f"Expected {len(rubric_batch)} results, got {len(eval_results) if isinstance(eval_results, list) else 'non-list'}"
                )

            normalized_results = []
            for idx, (result, rubric) in enumerate(zip(eval_results, rubric_batch)):
                if not isinstance(result, dict):
                    raise ValueError(
                        f"Result at position {idx} is {type(result).__name__}, expected object"
                    )
                if "criteria_met" not in result or "explanation" not in result:
                    raise ValueError(
                        f"Missing rubric fields in result at position {idx}: {result}"
                    )
                normalized_results.append(
                    {
                        "criteria_met": result["criteria_met"],
                        "explanation": result["explanation"],
                        "rubric": rubric,
                    }
                )

            logger.debug(f"Eval results: {normalized_results}")
            return normalized_results

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
            prompt_template = (
                PROMPT_MULTI_TURN if self.use_multi_turn_prompts else PROMPT
            )
            prompt = prompt_template.replace(
                "<<input_prompt>>", conversation_json
            ).replace("<<rubric>>", rubric.model_dump_json())

            eval_input = DataLoaderIterable(
                input=prompt,
                system_prompt=self._JUDGE_SYSTEM_PROMPT,
            )

            response = self.model.run([eval_input])[0]

            try:
                eval_result = json.loads(response)
                if not isinstance(eval_result, dict):
                    raise ValueError(
                        f"Expected JSON object with rubric fields, received {type(eval_result).__name__}"
                    )
                results.append(
                    {
                        "criteria_met": eval_result["criteria_met"],
                        "explanation": eval_result["explanation"],
                        "rubric": rubric,
                    }
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                results.append(
                    {
                        "criteria_met": False,
                        "explanation": f"Failed to parse response: {response}",
                        "rubric": rubric,
                    }
                )

        return results

    def _prepare_conversation(self, prediction, sample) -> Dict[str, Any]:
        """Append the model output (or parse a pass-through conversation) and
        return the conversation JSON plus any dataset-provided prompt parts.

        Mutates ``sample`` exactly once — done sequentially before the thread
        pool so judge_hashes can be computed and looked up in one batch.
        """
        if self.use_serialized_conversations:
            try:
                prediction_data = json.loads(prediction)
                sample.conversation = Conversation.model_validate(prediction_data)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.error(f"Failed to parse serialized conversation: {e}")
                raise ValueError(
                    f"use_serialized_conversations=True but prediction is not a valid Conversation JSON: {e}"
                )
        else:
            sample.conversation.conversation_turns.append(
                ConversationTurn(content=prediction, role="assistant")
            )

        return {
            "conversation_json": sample.conversation.model_dump_json(),
            "rubric_instructions": getattr(sample, "rubric_instructions", None),
            "rubric_definitions": getattr(sample, "rubric_definitions", None),
            "rubric_output_guidelines": getattr(
                sample, "rubric_output_guidelines", None
            ),
        }

    def _build_judge_input(
        self, conv: Dict[str, Any], sample_rubrics
    ) -> Dict[str, Any]:
        """Everything that determines the judge output for one sample. Hashed
        with the judge model config to key the rubric_outputs cache."""
        return {
            "conversation": conv["conversation_json"],
            "rubrics": [rubric.model_dump() for rubric in sample_rubrics],
            "multi_turn": self.use_multi_turn_prompts,
            "structured": bool(
                conv["rubric_instructions"]
                and conv["rubric_definitions"]
                and conv["rubric_output_guidelines"]
            ),
        }

    def _build_eval_specs(self, conv, sample_rubrics) -> List["_EvalSpec"]:
        """Assemble the judge input(s) for one sample, chunked by batch_size.

        Each spec's ``eval_input`` is hashed for the cache key and run directly
        on a miss, so the prompt is assembled exactly once.
        """
        specs = []
        for i in range(0, len(sample_rubrics), self.batch_size):
            rubric_batch = sample_rubrics[i : i + self.batch_size]
            eval_input, use_structured, is_single = self._build_eval_input(
                conv["conversation_json"],
                rubric_batch,
                conv["rubric_instructions"],
                conv["rubric_definitions"],
                conv["rubric_output_guidelines"],
            )
            specs.append(_EvalSpec(eval_input, rubric_batch, use_structured, is_single))
        return specs

    def _score_conversation(self, prepared, sample_rubrics, cached=None):
        """Score one conversation. On a cache hit, reconstruct from the stored
        judge output and skip the LLM entirely; otherwise run the prebuilt
        judge inputs."""
        conv, judge_input, specs = prepared

        if cached is not None:
            reconstructed = self._reconstruct_from_cache(cached, sample_rubrics)
            if reconstructed is not None:
                reconstructed["judge_input"] = judge_input
                reconstructed["from_cache"] = True
                return reconstructed

        grading_responses = []
        raw_responses = []
        for spec in specs:
            batch_results, batch_raw = self._run_eval_input(
                spec.eval_input,
                spec.rubric_batch,
                spec.use_structured,
                spec.is_single,
                conv["conversation_json"],
            )
            grading_responses.extend(batch_results)
            if batch_raw:
                raw_responses.append(batch_raw)

        question_score = self.calculate_score(sample_rubrics, grading_responses)

        return {
            "rubric_evaluations": grading_responses,
            "question_score": question_score,
            "judge_input": judge_input,
            "raw_response": "\n\n".join(raw_responses),
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

    def _aggregate_results(
        self, question_results: List[Dict]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Aggregate results across all questions.

        Args:
            question_results: List of question-level results

        Returns:
            Tuple containing aggregated metrics and serialized per-sample outputs
        """
        serialized_per_sample: List[Dict[str, Any]] = []
        for idx, result in enumerate(question_results):
            if not result:
                serialized_per_sample.append(
                    {
                        "question_index": idx,
                        "question_score": None,
                        "rubric_evaluations": [],
                    }
                )
                continue

            serialized_evaluations = [
                self._serialize_rubric_evaluation(evaluation)
                for evaluation in result.get("rubric_evaluations", [])
            ]

            serialized_per_sample.append(
                {
                    "question_index": idx,
                    "question_score": result.get("question_score"),
                    "rubric_evaluations": serialized_evaluations,
                }
            )

        # Filter out questions with None scores
        valid_scores = [
            result.get("question_score")
            for result in question_results
            if result and result.get("question_score") is not None
        ]

        if not valid_scores:
            summary = {
                "overall_score": 0.0,
                "num_questions": len(question_results),
                "num_valid_questions": 0,
            }
            return summary, serialized_per_sample

        # Calculate overall metrics
        overall_score = np.mean(valid_scores)
        std_dev = np.std(valid_scores, ddof=1) if len(valid_scores) > 1 else 0.0

        # Calculate bootstrap standard error (simplified)
        bootstrap_std = (
            std_dev / np.sqrt(len(valid_scores)) if len(valid_scores) > 0 else 0.0
        )

        # Aggregate by tags if available
        tag_scores = self._aggregate_by_tags(question_results)

        # Note: per-sample question_results are stored in _rubric_evaluation_results
        # and returned via rubric_evaluation_details. They are intentionally excluded
        # here to keep the run-level metric payload within DynamoDB's 400KB item limit.
        summary = {
            "overall_score": float(overall_score),
            "std_dev": float(std_dev),
            "bootstrap_std": float(bootstrap_std),
            "num_questions": len(question_results),
            "num_valid_questions": len(valid_scores),
            "tag_scores": tag_scores,
        }

        return summary, serialized_per_sample

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
            if not result or result.get("question_score") is None:
                continue

            for evaluation in result["rubric_evaluations"]:
                rubric = evaluation["rubric"]
                for tag in rubric.tags:
                    if tag not in tag_scores:
                        tag_scores[tag] = []

                    # Skip N/A rubrics (criteria_met=None) — same as calculate_score
                    # which excludes points=0.0 from the denominator.
                    # Counting None as 0 would inflate fail rates.
                    if evaluation["criteria_met"] is None:
                        continue
                    tag_scores[tag].append(1.0 if evaluation["criteria_met"] else 0.0)

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

    def _persist_rubric_results(
        self,
        *,
        cache_manager,
        dataset_name: str,
        question_results: List[Dict[str, Any]],
        serialized_samples: List[Any],
    ) -> None:
        """Persist per-question rubric evaluations for downstream analysis."""

        if not question_results:
            return

        metric_metadata = self._build_metric_metadata()
        records = []
        for index, (result, serialized_sample) in enumerate(
            zip(question_results, serialized_samples, strict=False)
        ):
            if result is None or serialized_sample is None:
                continue

            identifiers = cache_manager.get_cache_identifiers_for_sample(
                serialized_sample
            )

            rubric_evaluations = [
                self._serialize_rubric_evaluation(evaluation)
                for evaluation in result.get("rubric_evaluations", [])
            ]

            record = {
                "cache_key": identifiers["cache_key"],
                "dataset_row_hash": identifiers["dataset_row_hash"],
                "dataset_name": dataset_name,
                "config_hash": cache_manager.config_hash,
                "metric_name": self.metric_name,
                "question_index": index,
                "question_score": result.get("question_score"),
                "rubric_evaluations": rubric_evaluations,
                "metric_metadata": metric_metadata,
            }
            records.append(record)

        if records:
            cache_manager.batch_save_rubric_question_results(records)

    def _reconstruct_from_cache(
        self, cached: Dict[str, Any], sample_rubrics
    ) -> Optional[Dict[str, Any]]:
        """Rebuild grading responses from a cached judge row. Returns None on a
        shape mismatch so the caller re-runs the judge instead of mis-pairing.

        The score is recomputed from the cached judge decisions so scoring
        changes apply to reused judgements without a new judge call.
        """
        criteria = cached.get("criteria") or []
        if len(criteria) != len(sample_rubrics):
            return None

        grading_responses = [
            {
                "criteria_met": crit.get("criteria_met"),
                "explanation": crit.get("explanation"),
                "rubric": rubric,
            }
            for crit, rubric in zip(criteria, sample_rubrics)
        ]
        return {
            "rubric_evaluations": grading_responses,
            "question_score": self.calculate_score(
                sample_rubrics, grading_responses
            ),
            "raw_response": cached.get("raw_response"),
        }

    def _build_judge_model_config(self) -> Dict[str, Any]:
        """Whitelisted judge identity used in the judge_hash.

        Only stable identity (provider + model id) — generation defaults live
        inside the judge model and are not part of the key.
        """
        model_identifier = None
        if getattr(self, "model", None) is not None:
            model_identifier = getattr(
                self.model,
                "model_id",
                getattr(self.model, "model_name_or_path", None),
            )
        return {"provider": self.provider, "model_id": model_identifier}

    def _persist_rubric_outputs(
        self,
        *,
        cache_manager,
        question_results: List[Dict[str, Any]],
        serialized_samples: List[Any],
    ) -> None:
        """Write one content-addressed rubric_outputs row per judged sample."""

        if not question_results:
            return

        judge_model_config = self._build_judge_model_config()
        judge_model = judge_model_config.get("model_id")
        records = []
        for index, (result, serialized_sample) in enumerate(
            zip(question_results, serialized_samples, strict=False)
        ):
            if result is None or serialized_sample is None:
                continue
            if result.get("from_cache"):
                continue
            judge_hash = result.get("judge_hash")
            if not judge_hash:
                continue

            identifiers = cache_manager.get_cache_identifiers_for_sample(
                serialized_sample
            )
            criteria = [
                self._serialize_rubric_evaluation(evaluation)
                for evaluation in result.get("rubric_evaluations", [])
            ]

            records.append(
                {
                    "judge_hash": judge_hash,
                    "cache_key": identifiers["cache_key"],
                    "metric_name": self.metric_name,
                    "question_index": index,
                    "judge_model": judge_model,
                    "judge_model_config": judge_model_config,
                    "judge_input": result.get("judge_input"),
                    "question_score": result.get("question_score"),
                    "criteria": criteria,
                    "raw_response": result.get("raw_response"),
                    "error": result.get("error"),
                }
            )

        if records:
            cache_manager.save_rubric_outputs(records)

    @staticmethod
    def _serialize_rubric_evaluation(evaluation: Dict[str, Any]) -> Dict[str, Any]:
        rubric = evaluation.get("rubric")
        rubric_payload = (
            rubric.model_dump()
            if hasattr(rubric, "model_dump")
            else {
                "criterion": getattr(rubric, "criterion", None),
                "points": getattr(rubric, "points", None),
                "tags": getattr(rubric, "tags", None),
                "rubric_id": getattr(rubric, "rubric_id", None),
            }
        )

        return {
            "criteria_met": evaluation.get("criteria_met"),
            "explanation": evaluation.get("explanation"),
            "rubric": rubric_payload,
        }

    def _build_metric_metadata(self) -> Dict[str, Any]:
        """Assemble metadata describing the rubric evaluator configuration."""

        model_identifier = None
        if hasattr(self, "model") and self.model is not None:
            model_identifier = getattr(
                self.model,
                "model_id",
                getattr(self.model, "model_name_or_path", None),
            )

        metadata: Dict[str, Any] = {
            "provider": self.provider,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
        }

        if model_identifier:
            metadata["model_identifier"] = model_identifier
        if hasattr(self, "model") and self.model is not None:
            metadata["model_class"] = self.model.__class__.__name__
        if getattr(self, "metric_args", None):
            metadata["extra_args"] = self.metric_args

        return metadata
