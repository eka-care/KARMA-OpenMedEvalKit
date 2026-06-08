"""
Indian Drug MCQA dataset implementation.

Evaluates knowledge of Indian branded medications by testing the ability
to identify generic names/salt compositions from brand names.
Loaded from HuggingFace: ekacare/indian_drug_mcqa
"""

import logging
from typing import Any, Dict, Tuple

import pandas as pd

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset

logger = logging.getLogger(__name__)

CONFINEMENT_INSTRUCTIONS = """Instructions: The following are multiple choice questions about Indian pharmaceutical medications. Solve them in a step-by-step fashion. Output only the final answer (e.g., A, B, C, D, or E) at the end using the format "Final Answer: (X)". Question: <QUESTION>"""
SPLIT = "test"
DATASET_NAME = "ekacare/indian_drug_mcqa"


@register_dataset(
    dataset_name=DATASET_NAME,
    split=SPLIT,
    metrics=["exact_match"],
    task_type="mcqa",
)
class IndianDrugMCQADataset(BaseMultimodalDataset):
    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        confinement_instructions: str = CONFINEMENT_INSTRUCTIONS,
        **kwargs,
    ):
        self.confinement_instructions = confinement_instructions
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            confinement_instructions=confinement_instructions,
            **kwargs,
        )

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        all_options = [
            ("A", sample["option_a"]),
            ("B", sample["option_b"]),
            ("C", sample["option_c"]),
            ("D", sample["option_d"]),
            ("E", sample["option_e"]),
        ]
        # Filter out None/nan options
        valid_options = [
            (letter, opt)
            for letter, opt in all_options
            if opt is not None and (not isinstance(opt, float) or not pd.isna(opt))
        ]

        input_text = self._format_question(sample["question"], valid_options)
        prompt = self.confinement_instructions.replace("<QUESTION>", input_text)

        return DataLoaderIterable(
            input=prompt,
            expected_output=sample["correct_answer"],
        )

    def extract_prediction(self, response: str) -> Tuple[str, bool]:
        answer, success = "", False
        # Try "Final Answer:" format first
        if "Final Answer:" in response:
            raw = response.split("Final Answer:")[1].strip()
        else:
            raw = response.strip()
        # Extract first valid letter (A-E), handles: "A", "A**", "**A**", "(A)", "A.", etc.
        raw = raw.strip("()*")
        if raw and raw[0].upper() in "ABCDE":
            answer = raw[0].upper()
            success = True
        if not success:
            logger.warning(f"No answer found in response: {response}")
        return answer, success

    def _format_question(self, question: str, options: list) -> str:
        formatted = [f"{letter}. {opt}" for letter, opt in options]
        return f"{question}\n" + "\n".join(formatted)
