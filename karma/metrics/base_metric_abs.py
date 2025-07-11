from abc import ABC, abstractmethod


class BaseMetric(ABC):
    def __init__(self, metric_name: str, **kwargs):
        self.metric_name = metric_name
        self.metric_args = kwargs

    @abstractmethod
    def evaluate(self, predictions, references, rubrics, **kwargs):
        pass
