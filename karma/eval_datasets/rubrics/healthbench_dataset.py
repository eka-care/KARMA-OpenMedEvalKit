from karma.eval_datasets.rubrics.rubric_base_dataset import RubricBaseDataset
from karma.registries.dataset_registry import register_dataset

DATASET_NAME = "Tonic/Health-Bench-Eval-OSS-2025-07"
SPLIT = "oss_eval"
COMMIT_HASH = "0865a52cdf7ed7eff9923fe0dca419d9a0d6acbf"


@register_dataset(
    DATASET_NAME,
    split=SPLIT,
    commit_hash=COMMIT_HASH,
    metrics=["rubric_evaluation"],
    task_type="rubric_evaluation",
    default_args={"system_prompt": "healthbench_system"},
)
class HealthBenchDataset(RubricBaseDataset):
    def __init__(self, **kwargs):
        kwargs.pop("dataset_name", None)
        kwargs.pop("split", None)
        super().__init__(
            dataset_name=DATASET_NAME,
            split=SPLIT,
            **kwargs,
        )
