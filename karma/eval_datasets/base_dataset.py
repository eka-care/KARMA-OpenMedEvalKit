"""
Base dataset class for multimodal benchmarking.

This module provides the base interface that eval_datasets should implement
to provide model inputs directly to the benchmark system.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Generator, Optional, List
from torch.utils.data import IterableDataset
from datasets import load_dataset

logger = logging.getLogger(__name__)


class BaseMultimodalDataset(IterableDataset, ABC):
    """
    Base class for multimodal eval_datasets.

    Datasets should inherit from this class and implement the required methods
    to provide model inputs in the correct format for benchmarking.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        config: Optional[str] = None,
        stream: bool = True,
        commit_hash: Optional[str] = None,
        processors=None,
        max_samples: Optional[int] = None,
        confinement_instructions: str = "",
        **kwargs,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_name: Name of the dataset
            split: Split of the dataset
            config: Configuration/Subset of the dataset
            stream: Whether to stream the dataset
            commit_hash: Commit hash of the dataset
            **kwargs: Additional dataset-specific arguments
        """
        super().__init__()
        if processors is None:
            processors = []
        self.dataset_name = f"{dataset_name}_{config}" if config else dataset_name
        self.kwargs = kwargs
        self.config = config
        self.processors = processors
        # check if max samples is None then set it max integer
        self.max_samples = max_samples if max_samples is not None else float("inf")
        self.confinement_instructions = confinement_instructions
        self.split = split
        self.stream = stream
        self.commit_hash = commit_hash
        self.dataset = self.load_eval_dataset(
            dataset_name, split, config, stream, commit_hash
        )

    def load_eval_dataset(
        self,
        dataset_name: str,
        split: str = "test",
        config: Optional[str] = None,
        stream: bool = True,
        commit_hash: Optional[str] = None,
    ) -> IterableDataset:
        """Load the evaluation dataset."""
        if config:
            dataset = load_dataset(
                dataset_name,
                name=config,
                split=split,
                streaming=stream,
                revision=commit_hash,
            )
        else:
            dataset = load_dataset(
                dataset_name,
                split=split,
                streaming=stream,
                revision=commit_hash,
            )
        return dataset

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing the sample data including 'expected_output'
        """
        for idx, sample in enumerate(self.dataset):
            if idx >= self.max_samples:
                break
            item = self.format_item(sample)
            yield item
            # if isinstance(sample, dict):
            #     item = self.format_item(sample)
            #     # item["idx"] = idx
            #     yield item
            # else:
            #     # Handle non-dict samples appropriately
            #     item = self.format_item(dict(sample))
            #     # item["idx"] = idx
            #     yield item

    @abstractmethod
    def format_item(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a sample into a text prompt.

        Args:
            sample: A single sample from the dataset

        Returns:
            Processed sample
        """
        # Default implementation - should be overridden by subclasses
        pass

    def collate_fn(self, batch):
        """Simple collate function that preserves batch structure."""
        return batch

    def extract_prediction(self, prediction: str) -> Tuple[str, bool]:
        """
        Extract the prediction from the model output.
        """
        return prediction.strip(), True

    def postprocess(self, responses: List[str]) -> List[str]:
        """
        Postprocess the response.
        """
        if self.processors:
            for processor in self.processors:
                logger.info(
                    f"Running processor - {processor.name} for {self.dataset_name}"
                )
                responses = processor.process(responses)
        return responses


# def worker_init_fn(worker_id):
# """
# Worker initialization function for DataLoader with IterableDataset.
#
# This function splits the dataset workload across all workers to avoid duplicate data.
# """
# import math
# import torch
#
# worker_info = torch.utils.data.get_worker_info()
# dataset = worker_info.dataset  # the dataset copy in this worker process
#
# # These attributes must exist on your dataset
# overall_start = getattr(dataset, "start", None)
# overall_end = getattr(dataset, "end", None)
#
#     per_worker = int(
#         math.ceil((overall_end - overall_start) / float(worker_info.num_workers))
#     )
#     per_worker = int(
#         math.ceil((overall_end - overall_start) / float(worker_info.num_workers))
#     )
#     worker_id = worker_info.id
#     dataset.start = overall_start + worker_id * per_worker
#     dataset.end = min(dataset.start + per_worker, overall_end)
