"""
Calculator evaluation dataset.

Tests whether an LLM (with or without MCP-based medai tools) can correctly
compute numeric medical values such as BMI, Framingham risk, ANC, GFR, etc.
Data is loaded from HuggingFace with per-row tolerance for numeric comparison.
"""

import json
import logging
import re
from typing import Any, Dict, Tuple

from karma.data_models.dataloader_iterable import DataLoaderIterable, ToolPolicy
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset

logger = logging.getLogger(__name__)

DATASET_NAME = "ekacare/medical_calculator_eval"
SPLIT = "test"
CALCULATOR_TOOL_INSTRUCTION = (
    "When calculator tools are available, always use them to answer the question. "
    "Do not compute the result manually or from memory. "
    "Use the available calculator discovery and execution tools as needed, and return "
    "only the final answer."
)


@register_dataset(
    dataset_name=DATASET_NAME,
    split=SPLIT,
    metrics=["numeric_tolerance"],
    task_type="calculator",
    optional_args=["data_files"],
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

        # expected_output is stored as a JSON string in the dataset
        expected_output = sample["expected_output"]

        return DataLoaderIterable(
            input=prompt,
            expected_output=expected_output,
            other_args={
                "tolerance": sample.get("tolerance", 0.0),
                "primary_field": sample.get("primary_field", ""),
            },
            tool_policy=ToolPolicy(
                tool_instruction=CALCULATOR_TOOL_INSTRUCTION,
                first_turn_tool_choice="required",
                later_turn_tool_choice="auto",
            ),
        )

    def extract_prediction(self, prediction: str, **kwargs) -> Tuple[str, bool]:
        stripped = prediction.strip()

        # Try to extract a JSON object from the response (last one wins, as the
        # final answer appears at the end of tool-augmented responses)
        for match in reversed(list(re.finditer(r'\{[^{}]*\}', stripped))):
            try:
                parsed = json.loads(match.group())
                return json.dumps(parsed), True
            except json.JSONDecodeError:
                continue

        # Fall back: try direct float parse
        try:
            val = float(stripped)
            return str(val), True
        except ValueError:
            pass

        # Extract the last standalone number in the response — the final answer is at
        # the end, not the beginning (which may contain tool call IDs or intermediate
        # values). Use a negative lookbehind/lookahead to skip numbers attached to
        # letters (e.g. 'm2', 'kg/m2', 'call_4abc').
        matches = re.findall(r"(?<!\w)-?\d+\.?\d*(?!\w)", stripped)
        if matches:
            return matches[-1], True

        logger.warning(f"No value found in response: {prediction}")
        return "", False
