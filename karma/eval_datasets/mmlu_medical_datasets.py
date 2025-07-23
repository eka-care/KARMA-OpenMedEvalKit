"""
MMLU Medical dataset implementations with multimodal support.

This module provides multiple MMLU medical dataset classes that inherit from MedQA
and only update the dataset name, following the same optimization pattern.
"""

import logging

from karma.registries.dataset_registry import register_dataset
from karma.eval_datasets.medqa_dataset import MedQADataset

logger = logging.getLogger(__name__)

# Hardcoded confinement instructions - same as MedQA
CONFINEMENT_INSTRUCTIONS = """Instructions: The following are multiple choice questions about medical knowledge. Solve them in a
step-by-step fashion, starting by summarizing the available information. Output a single option from the
four options as the final answer. Question: <QUESTION> Response (think step by step and then
end with "Final Answer:" followed by *only* the letter corresponding to the correct answer enclosed in
parentheses)"""
SPLIT = "test"


@register_dataset(
    "openlifescienceai/mmlu_professional_medicine",
    commit_hash="0f2cda02673de66f90c7e1728e46d90590958700",
    split=SPLIT,
    metrics=["exact_match"],
    task_type="mcqa",
    optional_args=["confinement_instructions"],
)
class MMLUProfessionalMedicineDataset(MedQADataset):
    """MMLU Professional Medicine dataset inheriting from MedQA."""

    def __init__(self, dataset_name: str = "openlifescienceai/mmlu_professional_medicine", split: str = SPLIT, commit_hash: str = "0f2cda02673de66f90c7e1728e46d90590958700", confinement_instructions: str = CONFINEMENT_INSTRUCTIONS, **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            commit_hash=commit_hash,
            confinement_instructions=confinement_instructions,
            **kwargs,
        )
        self.dataset_name = "openlifescienceai/mmlu_professional_medicine"


@register_dataset(
    "openlifescienceai/mmlu_anatomy",
    split=SPLIT,
    commit_hash="a7a792bd0855aead8b6bf922fa22260eff160d6e",
    metrics=["exact_match"],
    task_type="mcqa",
    optional_args=["confinement_instructions"],
)
class MMLUAnatomyDataset(MedQADataset):
    """MMLU Anatomy dataset inheriting from MedQA."""

    def __init__(self, dataset_name: str = "openlifescienceai/mmlu_anatomy", split: str = SPLIT, commit_hash: str = "a7a792bd0855aead8b6bf922fa22260eff160d6e", confinement_instructions: str = CONFINEMENT_INSTRUCTIONS, **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            commit_hash=commit_hash,
            confinement_instructions=confinement_instructions,
            **kwargs,
        )


@register_dataset(
    "openlifescienceai/mmlu_college_biology",
    split=SPLIT,
    commit_hash="94b1278bb84c3005f90eef76d5846916f0d07f3a",
    metrics=["exact_match"],
    task_type="mcqa",
    optional_args=["confinement_instructions"],
)
class MMLUCollegeBiologyDataset(MedQADataset):
    """MMLU College Biology dataset inheriting from MedQA."""

    def __init__(self, dataset_name: str = "openlifescienceai/mmlu_college_biology", split: str = SPLIT, commit_hash: str = "94b1278bb84c3005f90eef76d5846916f0d07f3a", confinement_instructions: str = CONFINEMENT_INSTRUCTIONS, **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            commit_hash=commit_hash,
            confinement_instructions=confinement_instructions,
            **kwargs,
        )


@register_dataset(
    "openlifescienceai/mmlu_clinical_knowledge",
    split=SPLIT,
    commit_hash="e15116763fac9a86c1383c9d48428381b3335b22",
    metrics=["exact_match"],
    task_type="mcqa",
    optional_args=["confinement_instructions"],
)
class MMLUClinicalKnowledgeDataset(MedQADataset):
    """MMLU Clinical Knowledge dataset inheriting from MedQA."""

    def __init__(self, dataset_name: str = "openlifescienceai/mmlu_clinical_knowledge", split: str = SPLIT, commit_hash: str = "e15116763fac9a86c1383c9d48428381b3335b22", confinement_instructions: str = CONFINEMENT_INSTRUCTIONS, **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            commit_hash=commit_hash,
            confinement_instructions=confinement_instructions,
            **kwargs,
        )


@register_dataset(
    "openlifescienceai/mmlu_college_medicine",
    split=SPLIT,
    commit_hash="62ba72a3cc369ffec1def2a042f81ddc6837be12",
    metrics=["exact_match"],
    task_type="mcqa",
    optional_args=["confinement_instructions"],
)
class MMLUCollegeMedicineDataset(MedQADataset):
    """MMLU College Medicine dataset inheriting from MedQA."""

    def __init__(self, dataset_name: str = "openlifescienceai/mmlu_college_medicine", split: str = SPLIT, commit_hash: str = "62ba72a3cc369ffec1def2a042f81ddc6837be12", confinement_instructions: str = CONFINEMENT_INSTRUCTIONS, **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            commit_hash=commit_hash,
            confinement_instructions=confinement_instructions,
            **kwargs,
        )
