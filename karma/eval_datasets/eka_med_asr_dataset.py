from typing import Dict, Any
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset
from datasets import Audio

DATASET_NAME = "ekacare/eka-medical-asr-eval-dataset"
SPLIT = "test"
COMMIT_HASH = "5e20540ebd29868c8245e652412f27e40ca7acad"


@register_dataset(
    DATASET_NAME,
    metrics=["wer", "cer", "asr_semantic_metric"],
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    task_type="transcription",
    required_args=["language"],
    default_args={"language": "hi"},
    processors=["multilingual_text_processor"],
)
class EkaMedicalAsrDataset(BaseMultimodalDataset):
    def __init__(
        self,
        language: str = "hi",
        processors=None,
        **kwargs,
    ):
        """
        Initialize the EkaMedicalAsrDataset dataset.

        """
        super().__init__(
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
