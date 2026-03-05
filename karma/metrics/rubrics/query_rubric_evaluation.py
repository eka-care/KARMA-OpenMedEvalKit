"""
Query Rubric Evaluation Metric.

Evaluates retrieval query quality by filtering rubrics to those tagged
with 'stage:retrieval_query' and delegating to the base RubricMetric.
"""

import logging

from karma.metrics.rubrics.rubric_evaluation import RubricMetric
from karma.registries.metrics_registry import register_metric

logger = logging.getLogger(__name__)


@register_metric(
    name="query_rubric_evaluation",
    optional_args=["batch_size", "max_workers", "use_multi_turn_prompts", "use_serialized_conversations"],
    default_args={"batch_size": 100, "max_workers": 4, "use_multi_turn_prompts": False, "use_serialized_conversations": False},
)
class QueryRubricMetric(RubricMetric):
    """
    LLM-driven rubric evaluation metric filtered to stage:retrieval_query rubrics.
    """

    def evaluate(self, predictions, references=None, rubrics=None, **kwargs):
        # Filter each sample's rubrics to only stage:retrieval_query
        filtered_rubrics = []
        if rubrics:
            for sample_rubrics in rubrics:
                filtered = [
                    r for r in sample_rubrics
                    if "stage:retrieval_query" in (r.tags if hasattr(r, "tags") else [])
                ]
                filtered_rubrics.append(filtered)
        else:
            filtered_rubrics = rubrics

        result = super().evaluate(
            predictions=predictions,
            references=references,
            rubrics=filtered_rubrics,
            **kwargs,
        )

        # Rename keys to metric-specific names
        return {
            "query_rubric_evaluation": result.get("rubric_evaluation", {}),
            "query_rubric_evaluation_details": result.get("rubric_evaluation_details", []),
        }
