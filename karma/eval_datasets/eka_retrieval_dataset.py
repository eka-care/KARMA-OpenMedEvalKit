import logging
import os
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, Generator, Tuple
from datasets import load_dataset
from collections import defaultdict
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset

DATASET_NAME = "ekacare/Eka-IndicMTEB"
SPLIT="test"
COMMIT_HASH="ed3b481726fff03164311247fc352c21f260c1eb"

@register_dataset(
    dataset_name=DATASET_NAME,
    metrics=["ndcg@k", "recall@k", "map@k", "precision@k"],
    task_type="retrieval",
    optional_args=["language"],
    default_args={"split": "test", "language": "en"},
)
class EkaRetrievalDataset(BaseMultimodalDataset):
    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = "test",
        language: str = "en",
        **kwargs,
    ):
        super().__init__(
            dataset_name=dataset_name, split=split, config="queries", language=language, **kwargs
        )

        self.dataset_name = dataset_name
        self.split = split
        self.language = language
        self.hf_token = os.getenv("HF_TOKEN")
        self.logger = logging.getLogger(__name__)

        # Initialize dataset components
        self.corpus = []
        self.queries = {}
        self.qrels = {}
        self.dataset = None

    def load(
        self, split="test"
    ) -> tuple[list[dict], dict[str, str], dict[str, dict[str, int]]]:
        """Load corpus, queries, and qrels for the specified split."""
        self.logger.info("Loading Corpus...")
        self._load_corpus()
        self.logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
        self.logger.info("Doc Example: %s", self.corpus[0])

        self.logger.info("Loading Queries...")
        self._load_queries()
        self.logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
        self.logger.info("Query Example: %s", self.queries[next(iter(self.queries))])

        self.logger.info("Loading qrels...")
        self._load_qrels()
        self.logger.info("qrel Example: %s", self.qrels[next(iter(self.qrels))])

        return self.corpus, self.queries, self.qrels

    def _load_corpus(self):
        """Load the corpus data from HuggingFace dataset."""
        corpus_ds = load_dataset(
            self.dataset_name, "corpus", split=self.split, token=self.hf_token
        )

        corpus = []
        for row in corpus_ds:
            tmp = {
                "concept_id": row.get("concept_id"),
                "text": row.get("text"),
            }
            corpus.append(tmp)

        self.corpus = corpus

    def _load_queries(self):
        """Load the queries data from HuggingFace dataset."""
        queries_ds = load_dataset(
            self.dataset_name, "queries", split=self.split, token=self.hf_token
        )
        
        queries = {}
        for row in queries_ds:
            q_id, query = row.get("term_id"), row.get("term")
            
            queries[q_id] = query
        self.queries = queries

    def _load_qrels(self):
        """Load the qrels (relevance judgments) from HuggingFace dataset."""
        qrels_ds = load_dataset(
            self.dataset_name, "qrels", split=self.split, token=self.hf_token
        )

        qrels_dict = defaultdict(dict)

        for row in qrels_ds:
            qrels_dict[row.get("term_id")][str(row.get("concept_id"))] = row.get("score", 1)

        self.qrels = qrels_dict

    def _load_eval_dataset(self) -> Generator[Dict[str, Any], None, None]:
        """Load and yield evaluation examples."""
        if not self.corpus or not self.queries or not self.qrels:
            self.load(split=self.split)

        # Yield each query as an evaluation example
        for query_id, query_text in self.queries.items():
            yield {
                "id": str(query_id),
                "query": query_text,
                "qrels": {str(query_id): self.qrels.get(query_id, {})},
                "metadata": {"language": self.language, "split": self.split},
            }

    def __iter__(self) -> Generator[DataLoaderIterable, None, None]:
        """Iterate over the dataset and yield formatted examples."""
        if self.dataset is None:
            self.dataset = list(self._load_eval_dataset())
        
        for idx, sample in enumerate(self.dataset):
            if self.max_samples is not None and idx >= self.max_samples:
                break

            item = self.format_item(sample)
            yield item

    def format_item(self, item: Dict[str, Any]) -> DataLoaderIterable:
        """Format each item into DataLoaderIterable format for retrieval tasks."""
        query = item["query"]
        relevant_docs = item["qrels"]
        
        # Format the input for retrieval task
        input_text = f"{query}"

        return DataLoaderIterable(
            input=input_text,
            expected_output=relevant_docs,
            other_args={
                "query_id": str(item["id"]),
                "relevant_docs": relevant_docs,
                "task_type": "retrieval",
            },
        )

    def extract_prediction(self, prediction):
        return prediction, True

    def collate_fn(self, batch):
        return batch