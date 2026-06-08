"""
Query Rubric Evaluation Metric.

Scores tool/retrieval-stage rubrics (tagged ``stage:retrieval_query``) against
the *tool trace* the model produced — i.e. the queries it actually issued to
its tools — rather than the final assistant text.

Run-mode behavior:
- ``has_tools=False`` on the sample (no-tools run): the metric short-circuits
  and returns a null overall score with ``skipped_reason="has_tools=false"``.
  No grader LLM calls are made.
- ``has_tools=True`` (with-tools run): every sample is scored. Samples with an
  empty tool_trace (model failed to invoke any tool when it should have) get
  all-false grading responses synthesized directly — same numerical outcome
  as calling the grader on an empty string but with no API cost. They
  contribute a 0 to the overall mean.
"""

import logging
from typing import Dict, List

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
    LLM-driven rubric evaluation metric filtered to stage:retrieval_query rubrics,
    graded against the tool trace.
    """

    def evaluate(self, predictions, references=None, rubrics=None, **kwargs):
        samples = kwargs.get("samples") or []
        tool_traces: List[str] = kwargs.get("tool_traces") or [""] * len(predictions)

        # Decide run mode from the first sample's has_tools flag (uniform within
        # a single dataset run since the loader stamps every sample identically).
        has_tools = False
        if samples:
            other_args = getattr(samples[0], "other_args", None) or {}
            has_tools = bool(other_args.get("has_tools", False))

        if not has_tools:
            logger.info(
                "QueryRubricMetric short-circuiting: has_tools=false on dataset, "
                "query rubrics do not apply."
            )
            return {
                "query_rubric_evaluation": {
                    "overall_score": None,
                    "num_questions": len(predictions),
                    "num_valid_questions": 0,
                    "skipped_reason": "has_tools=false",
                },
                "query_rubric_evaluation_details": [],
            }

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

        # Build per-sample input_overrides (the tool trace) and precomputed
        # grading responses for samples where the model didn't actually issue
        # any retrieval queries (either no trace at all, or a trace that
        # contains only the model's direct answer with zero `[Tool Call:]`
        # markers — i.e. the model declined to use tools when it should have).
        input_overrides: List[str] = []
        precomputed: List[List[Dict] | None] = []
        for trace, rubric_list in zip(tool_traces, filtered_rubrics):
            trace = (trace or "").strip()
            n_tool_calls = trace.count("[Tool Call:") if trace else 0
            if n_tool_calls == 0:
                # Synthesize all-false responses; this counts as a 0 in the mean
                # without spending an LLM call. calculate_score will return
                # 0/total_possible_points = 0.0. Covers both fully-empty traces
                # and traces that only contain the model's direct answer.
                explanation = (
                    "Model declined to call any tool — final answer was given "
                    "directly without retrieval."
                    if trace
                    else "No tool trace recorded for this sample (model did not invoke any tool)."
                )
                precomputed.append(
                    [
                        {
                            "criteria_met": False,
                            "explanation": explanation,
                            "rubric": r,
                        }
                        for r in rubric_list
                    ]
                )
                input_overrides.append("")
            else:
                precomputed.append(None)
                input_overrides.append(trace)

        # Strip our consumed kwargs so the parent doesn't re-receive them and
        # forward the new override args.
        forwarded = {
            k: v
            for k, v in kwargs.items()
            if k not in {"tool_traces", "input_overrides", "precomputed_responses"}
        }
        result = super().evaluate(
            predictions=predictions,
            references=references,
            rubrics=filtered_rubrics,
            input_overrides=input_overrides,
            precomputed_responses=precomputed,
            **forwarded,
        )

        return {
            "query_rubric_evaluation": result.get("rubric_evaluation", {}),
            "query_rubric_evaluation_details": result.get("rubric_evaluation_details", []),
        }
