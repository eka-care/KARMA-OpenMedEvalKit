import torch
from typing import Dict, Any, Generator, Optional, List
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset
from karma.utils.audio import resample_audio
import numpy as np

DATASET_NAME = "ai4bharat/indicvoices_r"
SPLIT = "test"
COMMIT_HASH = "5f4495c91d500742a58d1be2ab07d77f73c0acf8"

@register_dataset(
    "indicvoices_r",
    metrics=["bleu", "wer", "cer"],
    task_type="transcription",
    required_args=["language"],
    default_args={"language": "hindi"},
    processors=["asr_wer_preprocessor"]
)
class IndicVoicesRDataset(BaseMultimodalDataset):
    def __init__(self, language: str = "hindi", dataset_name: str = DATASET_NAME , split: str = SPLIT, stream: bool = True, commit_hash: str = COMMIT_HASH, processors: Optional[List] = [], **kwargs):
        """
        Initialize the IndicVoicesR dataset.
        
        """
        super().__init__(dataset_name = dataset_name, config=language, split = split, stream = stream, commit_hash = commit_hash, processors=processors, **kwargs)
        self.language = language
        self.dataset_name = f"{DATASET_NAME}-{self.language}"


    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        audio_info = sample.get("audio",{})
        sampling_rate = audio_info.get("sampling_rate")
        
        audio_array = np.array(audio_info.get("array"), dtype=np.float32)
        if sampling_rate != 16000:
            audio_array = resample_audio(audio_array, orig_sr=sampling_rate, target_sr=16000)

        return DataLoaderIterable(
            audio=audio_array,
            expected_output=sample.get("text", ""),
            other_args={"language": sample.get("lang", "unknown")}
        )