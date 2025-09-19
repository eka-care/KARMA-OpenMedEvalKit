from typing import Dict, Any
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset
from datasets import Audio

DATASET_NAME = "ekacare/vistaar_small_asr_eval"
SPLIT = "test"
COMMIT_HASH = "8b7b3e8d11d2da441c6c708c2d9a4a7958b57881"


@register_dataset(
    DATASET_NAME,
    metrics=["wer", "cer"],
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    task_type="transcription",
    required_args=["language"],
    default_args={"language": "hi"},
    processors=["general_text_processor"],
)
class VistaarSmallDataset(BaseMultimodalDataset):
    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        commit_hash: str = COMMIT_HASH,
        language: str = "hi",
        processors=None,
        **kwargs,
    ):
        """
        Initialize the IndicVoicesR dataset.

        """
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            commit_hash=commit_hash,
            config=language,
            processors=processors,
            **kwargs,
        )
        self.language = language
        self.dataset_name = f"{DATASET_NAME}-{self.language}"
        self.dataset = self.dataset.cast_column(
            "audio", Audio(sampling_rate=16000, decode=False)
        )

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        return DataLoaderIterable(
            audio=sample.get("audio", {}).get("bytes"),
            expected_output=sample.get("text", ""),
            other_args={"language": sample.get("audio_language", "unknown")},
        )
