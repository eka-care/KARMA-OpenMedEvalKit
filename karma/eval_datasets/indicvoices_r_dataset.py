import torch
from typing import Dict, Any, Generator
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.utils.audio import resample_audio


DATASET_NAME = "ai4bharat/indicvoices_r"
SPLIT = "test"
COMMIT_HASH = "5f4495c91d500742a58d1be2ab07d77f73c0acf8"


class IndicVoicesRDataset(BaseMultimodalDataset):
    def __init__(self, language: str = "hi", dataset_name: str = DATASET_NAME , split: str = SPLIT, stream: bool = True, commit_hash: str = COMMIT_HASH, **kwargs):
        """
        Initialize the IndicVoicesR dataset.
        
        """
        super().__init__(dataset_name = dataset_name, language = language, split = split, stream = stream, commit_hash = commit_hash, **kwargs)
        self.language = language
        self.dataset_name = f"{DATASET_NAME}-{self.language}"

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        count = 0
        max_samples = self.kwargs.get("max_samples")
        
        for idx, sample in enumerate(self.dataset):
            if self.language and sample.get("lang") != self.language:
                continue

            item = self.format_item(sample)
            item["idx"] = idx
            yield item

            count += 1
            if max_samples and count >= max_samples:
                break


    def format_item(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        audio_info = sample.get("audio",{})
        sampling_rate = audio_info.get("sampling_rate")
        if not isinstance(audio_info.get("array"), torch.Tensor):
            audio_array = torch.tensor(audio_info.get("array"), dtype=torch.float32)
        if sampling_rate != 16000:
            audio_array = resample_audio(audio_array, orig_sr=sampling_rate, target_sr=16000)

        return {
            "audio": audio_array,
            "text": sample.get("text", ""),
            "language": sample.get("lang", "unknown")
        }