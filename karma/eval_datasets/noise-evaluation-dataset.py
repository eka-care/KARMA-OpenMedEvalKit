from typing import Dict, Any
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset
from datasets import Audio

DATASET_NAME = "ekacare/noise-evaluation-dataset"
SPLIT = "test"
COMMIT_HASH = "16b4cb019b72007a78aa66ed68451c9c7c6926f5"

@register_dataset(
    DATASET_NAME,
    metrics=["wer", "cer", "asr_semantic_metric"],
   commit_hash=COMMIT_HASH,
    split=SPLIT,
    task_type="transcription",
    default_args={"language": "en", "subset": "original"}
)
class NoiseEvaluationDataset(BaseMultimodalDataset):
    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        commit_hash: str = COMMIT_HASH,
        language: str = "en", 
        subset: str = "original",
        processors=None,
        **kwargs,
    ):
        """
        Initialize the NoiseEvaluationDataset dataset.
        
        Args:
            language: Language code (stored but not used as config)
            subset: Specific noise condition subset to evaluate on
        """
        super().__init__(
            dataset_name=DATASET_NAME,
            split=SPLIT,
            config=subset,  # Use subset as the config instead of language
            processors=processors,
            **kwargs,
        )
        self.language = language
        self.subset = subset
        self.dataset_name = f"{DATASET_NAME}-{self.subset}"
        self.dataset = self.dataset.cast_column(
            "audio", Audio(sampling_rate=16000, decode=False)
        )

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        return DataLoaderIterable(
            audio=sample.get("audio", {}).get("bytes"),
            expected_output=sample.get("text", ""),
            other_args={
                "language": sample.get("audio_language", "unknown"), 
                "recording_context": sample.get("recording_context", ""), 
                "type_concept": sample.get("type_concept", ""), 
                "entities": sample.get("medical_entities", []),
            },
        )
