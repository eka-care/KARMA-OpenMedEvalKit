from karma.eval_datasets.rubrics.rubric_base_dataset import RubricBaseDataset
from karma.registries.dataset_registry import register_dataset

base_default_system_prompt = """You are a helpful assistant"""

DATASET_NAME = "Tonic/Health-Bench-Eval-OSS-2025-07"
SPLIT = "oss_eval"
COMMIT_HASH = "0865a52cdf7ed7eff9923fe0dca419d9a0d6acbf"


@register_dataset(
    DATASET_NAME,
    split=SPLIT,
    commit_hash=COMMIT_HASH,
    metrics=["rubric_evaluation"],
    optional_args=["system_prompt"],
    task_type="rubric_evaluation",
)
class HealthBenchDataset(RubricBaseDataset):
    def __init__(self, system_prompt=base_default_system_prompt, **kwargs):
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.system_prompt = system_prompt
