import evaluate

from karma.metrics.base_metric_abs import BaseMetric
from karma.registries.metrics_registry import register_metric


class HfMetric(BaseMetric):
    def __init__(self, metric_name: str):
        super().__init__(metric_name)
        self.metric = evaluate.load(metric_name)

    def evaluate(self, predictions, references, **kwargs):
        return self.metric.compute(predictions=predictions, references=references)


@register_metric("bleu")
class BleuMetric(HfMetric):
    def __init__(self, metric_name: str = "bleu"):
        super().__init__(metric_name)

    def evaluate(self, predictions, references, smooth=True, **kwargs):
        references = [[ref] for ref in references]
        return self.metric.compute(
            predictions=predictions, references=references, smooth=smooth
        )


@register_metric("exact_match")
class ExactMatchMetric(HfMetric):
    def __init__(self, metric_name: str = "exact_match"):
        super().__init__(metric_name)


@register_metric("wer")
class WERMetric(HfMetric):
    def __init__(self, metric_name: str = "wer"):
        super().__init__(metric_name)

@register_metric("cer")
class CERMetric(HfMetric):
    def __init__(self, metric_name: str = "cer"):
        super().__init__(metric_name)