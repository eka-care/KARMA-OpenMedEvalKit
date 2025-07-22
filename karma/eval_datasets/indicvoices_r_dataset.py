from tkinter.constants import FALSE
import torch
from typing import Dict, Any, Generator, Optional, List
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset
from karma.utils.audio import resample_audio
import numpy as np
from datasets import Audio

DATASET_NAME = "ai4bharat/indicvoices_r"
SPLIT = "test"
COMMIT_HASH = "5f4495c91d500742a58d1be2ab07d77f73c0acf8"


@register_dataset(
    DATASET_NAME,
    metrics=["asr_semantic_metric"],  # metrics=["wer", "cer", "asr_semantic_metric"],
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
            "audio", Audio(sampling_rate=16000, decode=False)
        )

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        audio_info = sample.get("audio", {})
        audio_data = audio_info.get("bytes")
        
        return DataLoaderIterable(
            audio=audio_data,
            expected_output=sample.get("text", ""),
        )
