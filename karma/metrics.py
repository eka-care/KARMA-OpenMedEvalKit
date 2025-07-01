import evaluate
from karma.registries.metrics_registry import register_metric, BaseMetric


class HfMetric(BaseMetric):
    def __init__(self, metric_name: str):
        super().__init__(metric_name)
        self.metric = evaluate.load(metric_name)

    def evaluate(self, predictions, references):
        return self.metric.compute(predictions=predictions, references=references)


@register_metric("bleu")
class BleuMetric(HfMetric):
    def __init__(self, metric_name: str = "bleu"):
        super().__init__(metric_name)
    def evaluate(self, predictions, references, smooth: bool = True):
        references = [[ref] for ref in references]
        return self.metric.compute(
            predictions=predictions, references=references, smooth=smooth
        )


@register_metric("exact_match")
class ExactMatchMetric(HfMetric):
    def __init__(self, metric_name: str = "exact_match"):
        super().__init__(metric_name)
