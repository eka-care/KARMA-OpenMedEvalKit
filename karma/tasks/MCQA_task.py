from typing import Dict

from datasets import Dataset
from transformers import AutoModel


class MCQATask:
    """
    Multiple Choice Questions (MCQA) - based dataset metric computation.
    ALl the MCQA tasks follow exact_match metric.
    Supported datasets for this task:
        - openlifescienceai/pubmedqa
        - openlifescienceai/medmcqa
        - openlifescienceai/medqa (This is also mcq)
        - openlifescienceai/mmlu_professional_medicine
        - openlifescienceai/mmlu_anatomy
        - openlifescienceai/mmlu_college_biology
        - openlifescienceai/mmlu_clinical_knowledge
        - openlifescienceai/mmlu_college_medicine
    """

    def __init__(self, model: AutoModel, dataset: Dataset):
        self.dataset = dataset

    def compute_metrics(self, metric: Dict):
        pass

    def batch_predict(self, **kwargs):
        pass

    def evaluate(self, **kwargs):
        pass
