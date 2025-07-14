import json
import numpy as np
import logging

from typing import List, Dict, Any, Optional
from karma.metrics.base_metric_abs import BaseMetric
from karma.models.openai_llm import OpenAILLM
from karma.registries.metrics_registry import register_metric
from karma.data_models.dataloader_iterable import DataLoaderIterable

logger = logging.getLogger(__name__)


@register_metric(
    name="medical_records_rubric_evaluation",
    required_args=["provider_to_use", "model_id"],
    default_args={"provider_to_use": "openai", "model_id": "gpt-4o-mini"}
)
class MedicalRecordsRubricMetric(BaseMetric):
    """
    LLM driven rubric evaluation metric for medical records parsing.
    
    Evaluates JSON outputs generated from medical record images against
    structured rubrics. Unlike HealthBench which loops through individual 
    criteria, this metric sends the entire rubric prompt to the judge LLM 
    and expects all rubric scores in a single response.
    """

    def __init__(self, metric_name: str, provider_to_use: str = "openai", model_id: str = "gpt-4o-mini", **kwargs):
        super().__init__(metric_name=metric_name, **kwargs)
        self.provider = provider_to_use
        logger.info(f"Got {provider_to_use} rubric evaluation metric for medical records parsing")
        if self.provider == "openai":
            self.model = OpenAILLM(model_name_or_path=model_id)

    def evaluate(self, predictions, references=None, rubrics=None, **kwargs):
        """
        Evaluate JSON predictions from medical image parsing against rubrics using LLM-based scoring.

        Args:
            predictions: List of JSON responses from multimodal model parsing medical images
            references: Not used in rubric evaluation  
            rubrics: Not used - rubrics are embedded in sample other_args
            **kwargs: Additional arguments including 'samples' with rubric prompts

        Returns:
            Dict containing evaluation results
        """
        question_results = []
        samples = kwargs["samples"]
        logger.info(
            f"Evaluating {len(predictions)} JSON outputs with {self.provider} model - {self.model}"
        )
        
        for prediction, sample in zip(predictions, samples):
            # Get the rubric prompt from the sample's other_args
            rubric_prompt = sample.other_args.get("rubric_prompt", "")
            if not rubric_prompt:
                logger.error("No rubric prompt found in sample.other_args['rubric_prompt']")
                continue
            
            # Create the complete evaluation prompt by appending the JSON output
            complete_prompt = f"{rubric_prompt}\n\nJSON OUTPUT TO EVALUATE:\n{prediction}"
            
            # Create evaluation input
            eval_input = DataLoaderIterable(
                input=complete_prompt,
                system_prompt="You are an expert evaluator for medical records parsing tasks.",
            )
            logger.info("Running rubric evaluation on complete prompt")
            # Run model evaluation
            response = self.model.run([eval_input])[0]
            
            # Parse JSON response - handle markdown wrapping
            try:
                # Remove markdown code block wrapper if present
                clean_response = response.strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response[7:]  # Remove ```json
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]  # Remove ```
                clean_response = clean_response.strip()
                
                eval_result = json.loads(clean_response)
                logger.info(f"Evaluation result: {eval_result}")
                
                # Extract rubric scores
                rubric_scores = eval_result.get("rubric_scores", {})
                
                # Calculate score for this question
                question_score = self.calculate_score(rubric_scores)
                
                question_results.append({
                    "rubric_scores": rubric_scores,
                    "question_score": question_score,
                    "evaluation_response": response,
                })
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse response: {response}")
                question_results.append({
                    "rubric_scores": {},
                    "question_score": 0.0,
                    "evaluation_response": response,
                    "error": f"Failed to parse response: {response}",
                })

        # Aggregate results
        return {
            "medical_records_rubric_evaluation": self._aggregate_results(question_results)
        }

    def calculate_score(self, rubric_scores: Dict[str, int]) -> float:
        """
        Calculate the score for a single question based on rubric evaluations.

        Args:
            rubric_scores: Dictionary mapping rubric IDs to scores (0 or 1)

        Returns:
            Score as a float between 0 and 1
        """
        if not rubric_scores:
            return 0.0
            
        # For medical records parsing, each rubric criterion is worth 1 point
        # Score is the percentage of criteria met
        total_criteria = len(rubric_scores)
        met_criteria = sum(rubric_scores.values())
        
        overall_score = met_criteria / total_criteria if total_criteria > 0 else 0.0
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

        # Aggregate by individual rubric criteria across all samples
        rubric_breakdown = self._aggregate_by_rubric_id(question_results)

        return {
            "overall_score": float(overall_score),
            "std_dev": float(std_dev),
            "bootstrap_std": float(bootstrap_std),
            "num_questions": len(question_results),
            "num_valid_questions": len(valid_scores),
            "rubric_breakdown": rubric_breakdown,
            # Uncomment to include detailed results if needed
            # "question_results": question_results
        }

    def _aggregate_by_rubric_id(
        self, question_results: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate scores by individual rubric IDs.

        Args:
            question_results: List of question-level results

        Returns:
            Dict of rubric_id -> aggregated metrics
        """
        rubric_scores = {}

        for result in question_results:
            if "rubric_scores" not in result:
                continue

            for rubric_id, score in result["rubric_scores"].items():
                if rubric_id not in rubric_scores:
                    rubric_scores[rubric_id] = []
                rubric_scores[rubric_id].append(float(score))

        # Aggregate rubric scores
        aggregated_rubrics = {}
        for rubric_id, scores in rubric_scores.items():
            if scores:
                aggregated_rubrics[rubric_id] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                    "count": len(scores),
                    "pass_rate": float(np.mean([s for s in scores if s > 0])) if any(s > 0 for s in scores) else 0.0,
                }

        return aggregated_rubrics 