import evaluate
from sklearn.metrics import f1_score
from collections import Counter
import re
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


@register_metric("exact_match", optional_args=["ignore_case"], default_args={"ignore_case": True})
class ExactMatchMetric(HfMetric):
    def __init__(self, metric_name: str = "exact_match", **kwargs):
        super().__init__(metric_name)
    
    def evaluate(self, predictions, references, **kwargs):
        return self.metric.compute(predictions=predictions, references=references, ignore_case=kwargs.get("ignore_case", True))


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

@register_metric("tokenised_f1")
class TokenisedF1Metric(BaseMetric):
    def __init__(self, metric_name: str = "tokenised_f1", **kwargs):
        super().__init__(metric_name)

    def tokenize(self, text):
        text = text.replace('\n', '').replace('.', '')
        return re.findall(r'\w+', text.lower())
    
    def evaluate(self, predictions, references , **kwargs):
        f1_scores = []
        for prediction, reference in zip(predictions, references):
            pred_tokens = self.tokenize(prediction)
            gold_tokens = self.tokenize(reference)
    
            pred_counts = Counter(pred_tokens)
            gold_counts = Counter(gold_tokens)
    
            # Compute overlap
            common = pred_counts & gold_counts
            num_same = sum(common.values())
    
            if num_same == 0:
                f1_scores.append(0.0)
                continue
    
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gold_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)
        return sum(f1_scores)/len(f1_scores)

    
