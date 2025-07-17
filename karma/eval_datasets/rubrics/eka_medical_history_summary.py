import os

from karma.eval_datasets.rubrics.rubric_base_dataset import RubricBaseDataset
from karma.registries.dataset_registry import register_dataset

base_default_system_prompt = """You are a helpful assistant. Generate a succinct summary of the medical history for the last 6 months."""

DATASET_NAME = "ekacare/ekacare_medical_history_summarisation"
SPLIT = "test"
COMMIT_HASH = "9098354aaa37633264117f74212d6b80983d0a21"


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
        from huggingface_hub import login

        try:
            login(os.getenv("HUGGINGFACE_TOKEN"))
        except ValueError as e:
            raise e
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.system_prompt = system_prompt
