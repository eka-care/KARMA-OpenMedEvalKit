from typing import Dict, Any, Tuple
import logging

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.eval_datasets.extraction_utils import extract_wrapped_mcq_answer
from karma.registries.dataset_registry import register_dataset

logger = logging.getLogger(__name__) 

DATASET_NAME = "GBaker/MedQA-USMLE-4-options"
SPLIT = "test"
COMMIT_HASH = None

CONFINEMENT_INSTRUCTIONS = """Instructions: The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, starting by summarizing the available information. Output a single option from the four options as the final answer. Question: <QUESTION> Response (think step by step and then end with "Final Answer:" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)"""


@register_dataset(DATASET_NAME, commit_hash=COMMIT_HASH, split=SPLIT, metrics=["exact_match"], task_type="mcqa")
class MedQAUSMLEDataset(BaseMultimodalDataset):
    def __init__(self, dataset_name: str = DATASET_NAME, split: str = SPLIT, commit_hash: str = COMMIT_HASH, **kwargs):
        self.confinement_instructions = CONFINEMENT_INSTRUCTIONS
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            stream=False,
            confinement_instructions=self.confinement_instructions,
            commit_hash=commit_hash,
            **kwargs
        )
    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        question = sample["question"]
        options = sample["options"]

        formatted_choices = []
        for label in ["A", "B", "C", "D"]:
            if label in options:
                formatted_choices.append(f"{label}. {options[label]}")

        formatted_question = f"Question: {question}\n" + "\n".join(formatted_choices)

        prompt = self.confinement_instructions.replace("<QUESTION>", formatted_question)

        correct_answer = sample["answer_idx"]

        return DataLoaderIterable(
            input=prompt,
            expected_output=correct_answer,
        )

    def extract_prediction(self, response: str, **kwargs) -> Tuple[str, bool]:
        """Extract answer from model response."""
        answer, success = extract_wrapped_mcq_answer(response, valid_letters="ABCD")
        if not answer:
            logger.warning(f"No answer found in response: {response}")
        return answer, success
