"""
Diagnostic accuracy metric for MedCaseReasoning.

LLM-as-judge metric that scores semantic equivalence between a model's
predicted diagnosis and the ground-truth diagnosis. Paper methodology:
arXiv 2505.11733v2 (Prompt 7).

The judge model is looked up by registered model name via
``karma.registries.model_registry.get_model`` — this avoids a hardcoded
provider-dispatch ladder so any registered judge (OpenAI, Bedrock, and
future providers like Gemini added in later phases) works out of the box.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.metrics.base_metric_abs import BaseMetric
from karma.registries.metrics_registry import register_metric
from karma.registries.model_registry import get_model

logger = logging.getLogger(__name__)

# Paper Prompt 7 — y/n judge prompt for diagnostic equivalence.
DIAGNOSTIC_ACCURACY_JUDGE_PROMPT = (
    "Is our predicted diagnosis correct (y/n)? "
    "Predicted diagnosis: {predicted_diagnosis}, "
    "True diagnosis: {actual_diagnosis} "
    "Answer [y/n]."
)

DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_MAX_WORKERS = 20


def extract_answer_tag(response: str) -> str:
    """Extract diagnosis text from ``<answer>...</answer>`` tags.

    Falls back to the last non-empty line if no answer tag is present.
    """
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    return lines[-1] if lines else response.strip()


@register_metric(
    name="medcasereasoning_diagnostic_accuracy",
    optional_args=["judge_model", "max_workers"],
    default_args={
        "judge_model": DEFAULT_JUDGE_MODEL,
        "max_workers": DEFAULT_MAX_WORKERS,
    },
)
class MedCaseReasoningDiagnosticAccuracyMetric(BaseMetric):
    """LLM-as-judge diagnostic accuracy metric for MedCaseReasoning."""

    non_percentage_fields = frozenset({"num_questions", "num_correct"})

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

        # Look up the judge by registered model name. get_model() already
        # invokes load_model() on the returned instance, so it is ready to
        # run() out of the box.
        logger.info(
            f"Initialising MedCaseReasoning diagnostic-accuracy metric with "
            f"judge='{judge_model}', max_workers={max_workers}"
        )
        self.model = get_model(judge_model)

    def _judge_single(
        self, predicted_diagnosis: str, actual_diagnosis: str
    ) -> Dict[str, Any]:
        """Judge whether predicted diagnosis matches actual diagnosis."""
        prompt = DIAGNOSTIC_ACCURACY_JUDGE_PROMPT.format(
            predicted_diagnosis=predicted_diagnosis,
            actual_diagnosis=actual_diagnosis,
        )
        try:
            eval_input = DataLoaderIterable(
                input=prompt,
                system_prompt="You are a medical expert evaluating diagnostic accuracy.",
            )
            response = self.model.run([eval_input])[0]
            response_text = (response or "").strip()
            is_correct = response_text.lower().startswith("y")
            return {
                "correct": is_correct,
                "judge_response": response_text,
                "predicted_diagnosis": predicted_diagnosis,
                "actual_diagnosis": actual_diagnosis,
                "judge_prompt": prompt,
            }
        except Exception as e:
            logger.warning(f"Judge call failed: {e}")
            return {
                "correct": False,
                "judge_response": f"Error: {e}",
                "predicted_diagnosis": predicted_diagnosis,
                "actual_diagnosis": actual_diagnosis,
                "judge_prompt": prompt,
            }

    def evaluate(
        self, predictions, references=None, rubrics=None, **kwargs
    ) -> Dict[str, Any]:
        """Evaluate diagnostic accuracy using LLM-as-judge."""
        if not predictions:
            return {
                "medcasereasoning_diagnostic_accuracy": {
                    "overall_score": 0.0,
                    "num_questions": 0,
                    "num_correct": 0,
                },
                "medcasereasoning_diagnostic_accuracy_details": [],
            }

        references = references or [""] * len(predictions)

        judge_inputs: List[tuple] = []
        for prediction, reference in zip(predictions, references):
            predicted_diagnosis = extract_answer_tag(prediction)
            actual_diagnosis = reference or ""
            judge_inputs.append((predicted_diagnosis, actual_diagnosis))

        results: List[Dict[str, Any]] = [None] * len(judge_inputs)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self._judge_single, pred_diag, act_diag): i
                for i, (pred_diag, act_diag) in enumerate(judge_inputs)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Judge failed for sample {idx}: {e}")
                    pred_diag, act_diag = judge_inputs[idx]
                    results[idx] = {
                        "correct": False,
                        "judge_response": f"Error: {e}",
                        "predicted_diagnosis": pred_diag,
                        "actual_diagnosis": act_diag,
                        "judge_prompt": "",
                    }

        num_correct = sum(1 for r in results if r["correct"])
        overall_score = num_correct / len(results) if results else 0.0

        return {
            "medcasereasoning_diagnostic_accuracy": {
                "overall_score": overall_score,
                "num_questions": len(results),
                "num_correct": num_correct,
            },
            "medcasereasoning_diagnostic_accuracy_details": results,
        }
