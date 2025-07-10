import evaluate

from karma.metrics.base_metric_abs import BaseMetric
from karma.registries.metrics_registry import register_metric


class HfMetric(BaseMetric):
    def __init__(self, metric_name: str, **kwargs):
        super().__init__(metric_name)
        self.metric = evaluate.load(metric_name)

    def evaluate(self, predictions, references, **kwargs):
        return self.metric.compute(predictions=predictions, references=references)


@register_metric(
    name="bleu",
    optional_args=["max_order", "smooth"],
    default_args={"max_order": 4, "smooth": True},
)
class BleuMetric(HfMetric):
    def __init__(self, metric_name: str = "bleu", **kwargs):
        super().__init__(metric_name)

    def evaluate(self, predictions, references, **kwargs):
        smooth = kwargs.get("smooth", True)
        references = [[ref] for ref in references]
        return self.metric.compute(
            predictions=predictions, references=references, smooth=smooth
        )


@register_metric("exact_match")
class ExactMatchMetric(HfMetric):
    def __init__(self, metric_name: str = "exact_match", **kwargs):
        super().__init__(metric_name)


@register_metric("f1")
class F1Metric(HfMetric):
    def __init__(self, metric_name: str = "f1", **kwargs):
        super().__init__(metric_name)


@register_metric("wer")
class WERMetric(HfMetric):
    def __init__(self, metric_name: str = "wer", **kwargs):
        super().__init__(metric_name)


@register_metric("cer")
class CERMetric(HfMetric):
    def __init__(self, metric_name: str = "cer", **kwargs):
        super().__init__(metric_name)
