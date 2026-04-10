"""Numeric tolerance metric for calculator evaluation."""

import json
import logging
from typing import Any, Dict

from karma.metrics.base_metric_abs import BaseMetric
from karma.registries.metrics_registry import register_metric

logger = logging.getLogger(__name__)


@register_metric("numeric_tolerance")
class NumericToleranceMetric(BaseMetric):
    """Per-row numeric comparison with configurable tolerance from sample metadata.

    Both predictions and references may be JSON strings. When a ``primary_field``
    is present in the sample's ``other_args``, the corresponding key is extracted
    from each JSON object before numeric comparison.
    """

    def __init__(self, metric_name: str = "numeric_tolerance", **kwargs):
        super().__init__(metric_name, **kwargs)

    def _extract_primary_value(self, value: str, primary_field: str) -> str:
        """Return the primary field's value from a JSON string, or the raw string."""
        if not primary_field:
            return value
        try:
            parsed = json.loads(value)
            return str(parsed[primary_field])
        except (json.JSONDecodeError, KeyError, TypeError):
            return value

    def evaluate(self, predictions, references, rubrics=None, samples=None, **kwargs) -> Dict[str, Any]:
        if not predictions:
            return {"numeric_tolerance": 0.0}

        matches = 0
        total = len(predictions)

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            tolerance = 0.0
            primary_field = ""
            if samples and i < len(samples):
                other_args = getattr(samples[i], "other_args", None)
                if other_args:
                    tolerance = float(other_args.get("tolerance", 0.0))
                    primary_field = other_args.get("primary_field", "")

            pred_str = self._extract_primary_value(pred, primary_field)
            ref_str = self._extract_primary_value(ref, primary_field)

            try:
                pred_val = float(pred_str)
                ref_val = float(ref_str)
            except (ValueError, TypeError):
                logger.debug(f"Row {i}: cannot parse pred={pred_str!r} or ref={ref_str!r} as float")
                continue

            if abs(pred_val - ref_val) <= tolerance:
                matches += 1
            elif ref_val != 0 and abs(pred_val / 100 - ref_val) <= tolerance:
                # Model returned percentage, ground truth is 0-1 scale
                matches += 1
            elif ref_val != 0 and abs(pred_val * 100 - ref_val) <= tolerance:
                # Model returned 0-1 scale, ground truth is percentage
                matches += 1

        accuracy = matches / total if total > 0 else 0.0
        return {"numeric_tolerance": accuracy}
