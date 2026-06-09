"""
Stratified Accuracy Metrics

Analyze accuracy broken down by metadata fields (country, demographic mentions, etc.)
Run inference once, slice results multiple ways.

Usage:
    @register_dataset("my_dataset", metrics=["exact_match", "stratified_accuracy"])
"""

from collections import defaultdict
from typing import Any

from karma.metrics.base_metric_abs import BaseMetric
from karma.registries.metrics_registry import register_metric


def _extract_other_args(samples: list | None) -> list[dict]:
    """Extract other_args from samples list."""
    if not samples:
        return []
    result = []
    for sample in samples:
        if hasattr(sample, "other_args") and sample.other_args:
            result.append(sample.other_args)
        elif isinstance(sample, dict) and "other_args" in sample:
            result.append(sample["other_args"])
        else:
            result.append({})
    return result


@register_metric(
    "stratified_accuracy",
    optional_args=["stratify_by"],
    default_args={"stratify_by": ["country"]},
)
class StratifiedAccuracyMetric(BaseMetric):
    """
    Computes accuracy stratified by one or more metadata fields.

    Args:
        stratify_by: List of field names in other_args to stratify by.
                     Default: ["country"]
                     Options: "country", "mentions_gender", "mentions_age", "specialty", etc.

    Returns:
        - stratified_accuracy: Overall accuracy
        - accuracy_by_{field}: Accuracy breakdown for each stratification field
        - disparity_by_{field}: Max accuracy gap within each field
        - sample_counts_by_{field}: Sample counts per group
    """

    def __init__(self, metric_name: str = "stratified_accuracy", **kwargs):
        super().__init__(metric_name, **kwargs)
        stratify_by = kwargs.get("stratify_by", ["country"])
        # Handle both string and list inputs
        if isinstance(stratify_by, str):
            self.stratify_by = [stratify_by]
        else:
            self.stratify_by = stratify_by

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
        samples: list | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        other_args = _extract_other_args(samples)

        if not other_args:
            return {self.metric_name: 0.0, "error": "No metadata provided"}

        # Overall accuracy
        correct = sum(1 for p, r in zip(predictions, references) if p == r)
        overall_accuracy = correct / len(predictions) if predictions else 0

        result = {
            self.metric_name: round(overall_accuracy, 4),
            "overall_accuracy": round(overall_accuracy, 4),
            "total_samples": len(predictions),
        }

        # Stratify by each field
        for field in self.stratify_by:
            field_correct = defaultdict(int)
            field_total = defaultdict(int)

            for pred, ref, args in zip(predictions, references, other_args):
                value = args.get(field)
                if value is None:
                    continue

                # Convert float flags to readable labels
                if field.startswith("mentions_") and isinstance(value, float):
                    value = f"with_{field.replace('mentions_', '')}" if value == 1.0 else f"without_{field.replace('mentions_', '')}"

                field_total[value] += 1
                if pred == ref:
                    field_correct[value] += 1

            if not field_total:
                continue

            # Calculate accuracy per group
            accuracy_by_field = {}
            for value in sorted(field_total.keys(), key=str):
                acc = field_correct[value] / field_total[value]
                accuracy_by_field[str(value)] = round(acc, 4)

            # Calculate disparity
            accuracies = list(accuracy_by_field.values())
            disparity = max(accuracies) - min(accuracies) if len(accuracies) >= 2 else 0.0

            result[f"accuracy_by_{field}"] = accuracy_by_field
            result[f"disparity_by_{field}"] = round(disparity, 4)
            result[f"sample_counts_by_{field}"] = dict(field_total)

        return result


@register_metric(
    "geographic_accuracy",
    optional_args=[],
    default_args={},
)
class GeographicAccuracyMetric(StratifiedAccuracyMetric):
    """Accuracy stratified by country."""

    def __init__(self, metric_name: str = "geographic_accuracy", **kwargs):
        kwargs["stratify_by"] = ["country"]
        super().__init__(metric_name, **kwargs)


@register_metric(
    "demographic_accuracy",
    optional_args=[],
    default_args={},
)
class DemographicAccuracyMetric(StratifiedAccuracyMetric):
    """Accuracy stratified by demographic mentions (gender, age)."""

    def __init__(self, metric_name: str = "demographic_accuracy", **kwargs):
        kwargs["stratify_by"] = ["mentions_gender", "mentions_age"]
        super().__init__(metric_name, **kwargs)
