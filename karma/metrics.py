import evaluate
from karma.preprocessors.indiclang import DevanagariTransliterator
from karma.metrics_registry import register_metric, BaseMetric

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
        self.transliterator = DevanagariTransliterator()
        
    def evaluate(self, predictions, references, smooth: bool = True):
        # Process predictions (list of strings)
        predictions = [self.transliterator.process(text=pred) for pred in predictions]
        
        # Process references (list of list of strings)
        # Each inner list contains reference translations for the corresponding prediction
        references = [[self.transliterator.process(text=ref) for ref in ref_list] for ref_list in references]
        
        return self.metric.compute(predictions=predictions, references=references, smooth=smooth)

@register_metric("exact_match")
class ExactMatchMetric(HfMetric):
    def __init__(self, metric_name: str = "exact_match"):
        super().__init__(metric_name)
