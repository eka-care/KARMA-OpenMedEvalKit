from karma.metrics.base_metric_abs import BaseMetric
from karma.registries.metrics_registry import register_metric


@register_metric(name="rubric_evaluation")
class RubricMetric(BaseMetric):
    def __init__(self):
        super().__init__(metric_name="rubric_evaluation")

    def evaluate(self, predictions, references=None, rubrics=None, **kwargs):
        pass
