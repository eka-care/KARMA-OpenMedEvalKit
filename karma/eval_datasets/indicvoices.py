from typing import Dict, Any
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset
from datasets import Audio

DATASET_NAME = "ai4bharat/IndicVoices"
SPLIT = "valid"
COMMIT_HASH = "21fd45013ce5870e52d89d38b2cb88b834e02f8e"


@register_dataset(
    DATASET_NAME,
    metrics=["wer", "cer", "asr_semantic_metric"],
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    task_type="transcription",
    required_args=["language"],
    default_args={"language": "hindi"},
    processors=["multilingual_text_processor"],
)
class IndicVoicesRDataset(BaseMultimodalDataset):
    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        commit_hash: str = COMMIT_HASH,
        language: str = "hindi",
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
            "audio_filepath", Audio(sampling_rate=16000, decode=False)
        )

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        audio_info = sample.get("audio_filepath", {})
        audio_data = audio_info.get("bytes")

        return DataLoaderIterable(
            audio=audio_data,
            expected_output=sample.get("text", ""),
        )