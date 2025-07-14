from karma.eval_datasets.rubrics.rubric_base_dataset import RubricBaseDataset
from karma.registries.dataset_registry import register_dataset

base_default_system_prompt = """You are a helpful assistant. Generate a succinct summary of the medical history for the last 6 months."""

DATASET_NAME = "ekacare/ekacare_medical_history_summarisation"
SPLIT = "test"
COMMIT_HASH = "68ca9cb775fbcc51ff30729905a7ea028b5796dd"


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
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.system_prompt = system_prompt
