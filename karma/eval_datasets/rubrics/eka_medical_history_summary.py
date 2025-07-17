from datetime import datetime
import os

from karma.eval_datasets.rubrics.rubric_base_dataset import RubricBaseDataset
from karma.registries.dataset_registry import register_dataset

# base_default_system_prompt = f"""
# You are a medical copilot with expansive medical knowledge that should mimic an experienced medical professional (a doctor).
# You will never paraphrase or speculate information, always faithfully reproduce the evidence as reported in all the output.
# Always cross-reference information across different sections of the patient data to ensure consistency in reporting
# Today is {datetime.now().strftime("%Y-%m-%d")}. Compute all relative dates in comparison to today.
# All dates of the evidences are formatted as %Y-%m-%d.
# Generate a succinct summary of the medical history for the most recent 6 months for a doctor.
# """
base_default_system_prompt = """You are a helpful assistant. Generate a succinct summary of the medical history for the last 6 months."""

DATASET_NAME = "ekacare/ekacare_medical_history_summarisation"
SPLIT = "test"
COMMIT_HASH = "ef267abbf76bd032aaaf40053ad67e1e44488ada"


@register_dataset(
    DATASET_NAME,
    split=SPLIT,
    commit_hash=COMMIT_HASH,
    metrics=["rubric_evaluation"],
    optional_args=["system_prompt"],
    task_type="rubric_evaluation",
)
class EkaMedicalHistorySummary(RubricBaseDataset):
    def __init__(self, system_prompt=base_default_system_prompt, **kwargs):
        super().__init__(
            dataset_name=DATASET_NAME,
            split=SPLIT,
            system_prompt=system_prompt,
            **kwargs,
        )
        self.system_prompt = system_prompt
