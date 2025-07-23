import logging
from typing import Any, Dict, Tuple, Generator, Optional
import os
import pandas as pd
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset

logger = logging.getLogger(__name__)

CONFINEMENT_INSTRUCTIONS = """Instructions: The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion. Output only the final answer (e.g., A, B, C, or D) at the end using the format "Final Answer: (X)". Question: <QUESTION>"""
SPLIT = "test"
DATASET_NAME = "nfi_mcqa"


@register_dataset(
    dataset_name=DATASET_NAME,
    split=SPLIT,
    metrics=["exact_match"],
    task_type="mcqa",
)
class NFIMCQADataset(BaseMultimodalDataset):
    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        confinement_instructions: str = CONFINEMENT_INSTRUCTIONS,
        **kwargs,
    ):
        # Load local data first
        self.data_path = (
            "/Users/hardikchhallani/Downloads/nfi-questions-10-generics.parquet"
        )
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        self.df = pd.read_parquet(self.data_path)

        self.dataset = None
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            confinement_instructions=confinement_instructions,
            **kwargs,
        )

        self.dataset_name = dataset_name
        self.confinement_instructions = confinement_instructions
        self.split = SPLIT
        self.stream = False
        self.processors = kwargs.get("processors", [])
        self.max_samples = kwargs.get("max_samples", None)

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing the sample data including 'expected_output'
        """
        if self.dataset is None:
            self.dataset = list(self.load_eval_dataset())  # cache once
        for idx, sample in enumerate(self.dataset):
            if self.max_samples is not None and idx >= self.max_samples:
                break
            item = self.format_item(sample)
            yield item

    def __len__(self):
        """Override length to use our local data."""
        return len(self.df)

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        input_text = self._format_question(sample["data"])
        correct_answer = sample["data"]["ground_truth"]

        prompt = self.confinement_instructions.replace("<QUESTION>", input_text)

        # Create DataLoaderIterable with input field for OpenAI LLM
        dataloader_item = DataLoaderIterable(
            input=prompt, expected_output=correct_answer
        )

        # Ensure conversation is None so OpenAI LLM uses input field
        dataloader_item.conversation = None

        return dataloader_item

    def extract_prediction(self, response: str) -> Tuple[str, bool]:
        """
        Extracts the model's predicted option from its response.
        Expected format: Final Answer: (A)
        """
        answer, success = "", False
        if "Final Answer:" in response:
            answer = response.split("Final Answer:")[1].strip()
            if answer.startswith("(") and answer.endswith(")"):
                answer = answer[1:-1]
            success = True
        if not success:
            logger.warning(f"No answer found in response: {response}")
        return answer, success

    def _format_question(self, data: Dict[str, Any]) -> str:
        """
        Builds a multiple-choice formatted question.
        """
        question = data["question"]
        options = data["options"]
        letters = ["A", "B", "C", "D"]
        formatted = [f"{l}. {opt}" for l, opt in zip(letters, options)]
        return f"{question}\n" + "\n".join(formatted)

    def load_eval_dataset(self,
                          dataset_name: str,
                          split: str = "test",
                          config: Optional[str] = None,
                          stream: bool = True,
                          commit_hash: Optional[str] = None,
                          **kwargs):
        logger.info("Using custom local load method")
        logger.info(f"Loading local dataset from {self.data_path}")

        for _, row in self.df.iterrows():
            # Handle model_output_parsed safely
            prediction = None
            parsed_output = row.get("model_output_parsed", None)
            if isinstance(parsed_output, dict):
                prediction = parsed_output.get("prediction", None)

            yield {
                "id": row["index"],
                "data": {
                    "question": row["question"],
                    "options": row["options"],
                    "ground_truth": row["ground_truth"],
                },
                "prediction": prediction,
                "metadata": {
                    "generic_name": row.get("generic_name", None),
                    "category": row.get("category", None),
                    "citation": row.get("citation", None),
                },
            }
