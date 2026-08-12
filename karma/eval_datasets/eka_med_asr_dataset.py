from typing import Dict, Any
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset
from karma.utils.language import normalize_language, sample_language
from datasets import Audio

DATASET_NAME = "ekacare/eka-medical-asr-evaluation-dataset"
SPLIT = "test"
COMMIT_HASH = "991bc807cab1f323f0283c836c634796bbf1ed3e"


@register_dataset(
    DATASET_NAME,
    metrics=["wer", "cer"],
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    task_type="transcription",
    required_args=["subset"],
    default_args={"user_prompt": "asr_transcribe", "subset": "hi"},
)
class EkaMedicalAsrDataset(BaseMultimodalDataset):
    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        commit_hash: str = COMMIT_HASH,
        subset: str = "hi",
        processors=None,
        **kwargs,
    ):
        """
        Initialize the EkaMedicalAsrDataset dataset.

        """
        language = normalize_language(subset)
        super().__init__(
            dataset_name=DATASET_NAME,
            split=SPLIT,
            config=language,
            processors=processors,
            **kwargs,
        )
        self.language = language
        self.dataset_subset = language
        self.dataset = self.dataset.cast_column(
            "audio", Audio(sampling_rate=16000, decode=False)
        )

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        return DataLoaderIterable(
            audio=sample.get("audio", {}).get("bytes"),
            expected_output=sample.get("text", ""),
            language=sample_language(sample.get("audio_language")),
            other_args={
                "recording_context": sample.get("recording_context", ""),
                "type_concept": sample.get("type_concept", ""),
                "entities": sample.get("medical_entities", []),
            },
        )
