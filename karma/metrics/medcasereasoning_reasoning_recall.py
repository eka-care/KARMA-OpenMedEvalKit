"""
Reasoning recall metric for MedCaseReasoning.

Measures the fraction of ground-truth reasoning statements that the model's
``<think>``-block response covers, judged via LLM-as-judge per statement.
Paper methodology: arXiv 2505.11733v2.

The judge model is looked up by registered model name via
``karma.registries.model_registry.get_model`` — this avoids a hardcoded
provider-dispatch ladder so any registered judge (OpenAI, Bedrock, and
future providers like Gemini added in later phases) works out of the box.
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.metrics.base_metric_abs import BaseMetric
from karma.registries.metrics_registry import register_metric
from karma.registries.model_registry import get_model

logger = logging.getLogger(__name__)

REASONING_RECALL_JUDGE_PROMPT = (
    "You are evaluating whether a model's diagnostic reasoning covers a "
    "specific clinical reasoning point.\n\n"
    "Ground-truth reasoning statement:\n{statement}\n\n"
    "Model's reasoning response:\n{model_reasoning}\n\n"
    "Does the model's response cover the same clinical insight as the "
    "ground-truth statement, even if worded differently? Answer with only "
    "'y' or 'n'."
)

DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_MAX_WORKERS = 20


def extract_think_tag(response: str) -> str:
    """Extract reasoning content from ``<think>...</think>`` tags.

    Falls back to the entire response when the model omits think tags.
    """
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response.strip()


@register_metric(
    name="medcasereasoning_reasoning_recall",
    optional_args=["judge_model", "max_workers"],
    default_args={
        "judge_model": DEFAULT_JUDGE_MODEL,
        "max_workers": DEFAULT_MAX_WORKERS,
    },
)
class MedCaseReasoningReasoningRecallMetric(BaseMetric):
    """LLM-as-judge reasoning-recall metric for MedCaseReasoning."""

    non_percentage_fields = frozenset({"num_questions", "num_valid_questions"})

    def __init__(
        self,
        metric_name: str,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        max_workers: int = DEFAULT_MAX_WORKERS,
        **kwargs,
    ):
        super().__init__(metric_name=metric_name, **kwargs)
        if isinstance(max_workers, str):
            max_workers = int(max_workers)
        self.max_workers = max_workers
        self.judge_model_name = judge_model

        logger.info(
            f"Initialising MedCaseReasoning reasoning-recall metric with "
            f"judge='{judge_model}', max_workers={max_workers}"
        )
        self.model = get_model(judge_model)

    def _judge_single_step(
        self, statement: str, model_reasoning: str
    ) -> Dict[str, Any]:
        """Judge whether model reasoning covers a single ground-truth statement.

        Returns a dict with ``covered`` (bool), ``judge_prompt``, and
        ``judge_raw_response``.
        """
        prompt = REASONING_RECALL_JUDGE_PROMPT.format(
            statement=statement,
            model_reasoning=model_reasoning,
        )
        try:
            eval_input = DataLoaderIterable(
                input=prompt,
                system_prompt="You are an expert medical reasoning evaluator.",
            )
            response = self.model.run([eval_input])[0]
            response_text = (response or "").strip()
            is_covered = response_text.lower().startswith("y")
            return {
                "covered": is_covered,
                "judge_prompt": prompt,
                "judge_raw_response": response_text,
            }
        except Exception as e:
            logger.warning(f"Judge step failed: {e}")
            return {
                "covered": False,
                "judge_prompt": prompt,
                "judge_raw_response": f"Error: {e}",
            }

    def evaluate(
        self, predictions, references=None, rubrics=None, **kwargs
    ) -> Dict[str, Any]:
        """Evaluate reasoning recall using LLM-as-judge per ground-truth step."""
        samples = kwargs.get("samples", [])

        if not predictions:
            return {
                "medcasereasoning_reasoning_recall": {
                    "overall_score": 0.0,
                    "num_questions": 0,
                    "num_valid_questions": 0,
                },
                "medcasereasoning_reasoning_recall_details": [],
            }

        # Build flat job list: (sample_idx, step_idx, statement, model_reasoning).
        # This lets us drive a single ThreadPoolExecutor across ALL judge calls
        # instead of serialising on a per-sample level.
        jobs: List[Tuple[int, int, str, str]] = []
        sample_steps_cache: Dict[int, List[str]] = {}
        for i, prediction in enumerate(predictions):
            sample = samples[i] if i < len(samples) else None
            try:
                other_args = (sample.other_args or {}) if sample is not None else {}
                steps = json.loads(other_args.get("reasoning_steps", "[]"))
                if not isinstance(steps, list):
                    steps = []
            except (json.JSONDecodeError, AttributeError, TypeError):
                steps = []
            sample_steps_cache[i] = steps
            model_reasoning = extract_think_tag(prediction)
            for j, statement in enumerate(steps):
                jobs.append((i, j, statement, model_reasoning))

        # Execute all judge calls in one flat pool.
        step_results: Dict[Tuple[int, int], Dict[str, Any]] = {}
        if jobs:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_key = {
                    executor.submit(self._judge_single_step, stmt, reasoning): (si, sj)
                    for si, sj, stmt, reasoning in jobs
                }
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        step_results[key] = future.result()
                    except Exception as e:
                        logger.warning(f"Judge failed for {key}: {e}")
                        step_results[key] = {
                            "covered": False,
                            "judge_prompt": "",
                            "judge_raw_response": f"Error: {e}",
                        }

        # Reassemble per-sample results.
        per_sample_results: List[Dict[str, Any]] = []
        for i in range(len(predictions)):
            steps = sample_steps_cache.get(i, [])
            num_steps = len(steps)
            if num_steps == 0:
                per_sample_results.append(
                    {
                        "recall": None,
                        "num_steps": 0,
                        "num_covered": 0,
                        "step_results": [],
                    }
                )
                continue
            sample_step_dicts = [
                step_results.get(
                    (i, j),
                    {
                        "covered": False,
                        "judge_prompt": "",
                        "judge_raw_response": "",
                    },
                )
                for j in range(num_steps)
            ]
            num_covered = sum(1 for r in sample_step_dicts if r["covered"])
            recall = num_covered / num_steps
            per_sample_results.append(
                {
                    "recall": recall,
                    "num_steps": num_steps,
                    "num_covered": num_covered,
                    "step_results": [r["covered"] for r in sample_step_dicts],
                }
            )

        # Overall mean recall, excluding samples with 0 ground-truth steps.
        valid_recalls = [
            r["recall"] for r in per_sample_results if r["recall"] is not None
        ]
        num_valid = len(valid_recalls)
        mean_recall = sum(valid_recalls) / num_valid if num_valid > 0 else 0.0

        return {
            "medcasereasoning_reasoning_recall": {
                "overall_score": mean_recall,
                "num_questions": len(per_sample_results),
                "num_valid_questions": num_valid,
            },
            "medcasereasoning_reasoning_recall_details": per_sample_results,
        }
