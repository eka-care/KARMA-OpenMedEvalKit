"""
AfriMed-QA Harm Detection Dataset

This module provides the harm detection subset of AfriMed-QA.
For MCQ evaluation, use the main afrimedqa_dataset.py which registers:
- intronhealth/afrimedqa_v2 (Consumer Queries)
- intronhealth/afrimedqa_v2/mcq (MCQ with stratified metrics)

Source: intronhealth/afrimedqa_v2
"""

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset

# Dataset constants
AFRIMEDQA_DATASET_NAME = "intronhealth/afrimedqa_v2"
AFRIMEDQA_SPLIT = "train"



@register_dataset(
    "intronhealth/afrimedqa_v2:harm",
    metrics=["harm_detection_accuracy"],
    task_type="harm_detection",
    split=AFRIMEDQA_SPLIT,
    default_args={
        "user_prompt": "afrimedqa_harm_eval",
    },
)
class AfriMedQAHarmDataset(BaseMultimodalDataset):
    """
    AfriMed-QA dataset for harm detection evaluation.

    Uses human-annotated harm ratings to evaluate model safety.
    Only includes samples with rated_harmful annotations (65 total: 15 harmful, 50 safe).
    """

    def __init__(
        self,
        dataset_name: str = AFRIMEDQA_DATASET_NAME,
        split: str = AFRIMEDQA_SPLIT,
        **kwargs,
    ):
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            **kwargs,
        )

    def __iter__(self):
        """Iterate over samples with harm annotations."""
        count = 0
        for sample in self.dataset:
            if self.max_samples and count >= self.max_samples:
                break

            # Only include samples with harm ratings
            if sample.get("rated_harmful") is None:
                continue

            formatted = self._format_item(sample)
            if formatted:
                count += 1
                yield formatted

    def _format_item(self, sample: dict) -> DataLoaderIterable | None:
        """Format a sample for harm evaluation."""
        question = sample.get("question_clean") or sample.get("question")
        if not question:
            return None

        prompt = self.confinement_instructions.format(question=question)

        # Harm label: 1.0 = harmful, 0.0 = not harmful
        is_harmful = sample.get("rated_harmful", 0.0) == 1.0

        return DataLoaderIterable(
            input=prompt,
            expected_output="harmful" if is_harmful else "safe",
            other_args={
                "sample_id": sample.get("sample_id"),
                "country": sample.get("country"),
                "question_type": sample.get("question_type"),
                "rated_harmful": sample.get("rated_harmful"),
                "rated_bias": sample.get("rated_bias"),
                "is_harmful": is_harmful,
            },
        )

    def format_item(self, sample: dict) -> DataLoaderIterable:
        return self._format_item(sample)
