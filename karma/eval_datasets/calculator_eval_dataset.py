"""
Calculator evaluation dataset.

Tests whether an LLM (with or without MCP-based medai tools) can correctly
compute numeric medical values such as BMI, Framingham risk, ANC, GFR, etc.
Data is loaded from HuggingFace with per-row tolerance for numeric comparison.
"""

import logging
import re
from typing import Any, Dict, Tuple

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset

logger = logging.getLogger(__name__)

DATASET_NAME = "ekacare/medical_calculator_eval"
SPLIT = "test"


@register_dataset(
    dataset_name=DATASET_NAME,
    split=SPLIT,
    metrics=["numeric_tolerance"],
    task_type="calculator",
)
class CalculatorEvalDataset(BaseMultimodalDataset):
    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        **kwargs,
    ):
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            **kwargs,
        )

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        question = sample["question_text"]
        confinement = sample.get("confinement_instruction", "")
        prompt = f"{question}\n{confinement}".strip() if confinement else question

        return DataLoaderIterable(
            input=prompt,
            expected_output=str(sample["expected_output"]),
            other_args={"tolerance": sample.get("tolerance", 0.0)},
        )

    def extract_prediction(self, prediction: str) -> Tuple[str, bool]:
        # Try direct float parse
        stripped = prediction.strip()
        try:
            val = float(stripped)
            return str(val), True
        except ValueError:
            pass

        # Fallback: extract first number via regex
        match = re.search(r"-?\d+\.?\d*", stripped)
        if match:
            return match.group(), True

        logger.warning(f"No numeric value found in response: {prediction}")
        return "", False
