from abc import ABC, abstractmethod
from typing import FrozenSet


class BaseMetric(ABC):
    # Sub-fields in the run-level payload that are raw counts (not 0-1 ratios).
    # Override in subclasses to prevent the UI from rendering them as percentages.
    non_percentage_fields: FrozenSet[str] = frozenset()

    def __init__(self, metric_name: str, **kwargs):
        self.metric_name = metric_name
        self.metric_args = kwargs

    @abstractmethod
    def evaluate(self, predictions, references, rubrics, **kwargs):
        pass
