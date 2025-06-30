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

SPLIT = "test"


@register_dataset(
    "mmlu_professional_medicine", metrics=["exact_match"], task_type="mcqa"
)
class MMLUProfessionalMedicineDataset(MedQADataset):
    """MMLU Professional Medicine dataset inheriting from MedQA."""

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="openlifescienceai/mmlu_professional_medicine",
            split=SPLIT,
            commit_hash="0f2cda02673de66f90c7e1728e46d90590958700",
            **kwargs,
        )


@register_dataset("mmlu_anatomy", metrics=["exact_match"], task_type="mcqa")
class MMLUAnatomyDataset(MedQADataset):
    """MMLU Anatomy dataset inheriting from MedQA."""

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="openlifescienceai/mmlu_anatomy",
            split=SPLIT,
            commit_hash="a7a792bd0855aead8b6bf922fa22260eff160d6e",
            **kwargs,
        )


@register_dataset("mmlu_college_biology", metrics=["exact_match"], task_type="mcqa")
class MMLUCollegeBiologyDataset(MedQADataset):
    """MMLU College Biology dataset inheriting from MedQA."""

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="openlifescienceai/mmlu_college_biology",
            split=SPLIT,
            commit_hash="94b1278bb84c3005f90eef76d5846916f0d07f3a",
            **kwargs,
        )


@register_dataset("mmlu_clinical_knowledge", metrics=["exact_match"], task_type="mcqa")
class MMLUClinicalKnowledgeDataset(MedQADataset):
    """MMLU Clinical Knowledge dataset inheriting from MedQA."""

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="openlifescienceai/mmlu_clinical_knowledge",
            split=SPLIT,
            commit_hash="e15116763fac9a86c1383c9d48428381b3335b22",
            **kwargs,
        )


@register_dataset("mmlu_college_medicine", metrics=["exact_match"], task_type="mcqa")
class MMLUCollegeMedicineDataset(MedQADataset):
    """MMLU College Medicine dataset inheriting from MedQA."""

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="openlifescienceai/mmlu_college_medicine",
            split=SPLIT,
            commit_hash="62ba72a3cc369ffec1def2a042f81ddc6837be12",
            **kwargs,
        )
