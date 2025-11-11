import logging
import torch
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from karma.utils.retrieval.index.base_index import BaseIndex


class FAISSIndex(BaseIndex):
    def __init__(
        self, score_function: str, logger: logging.Logger, save_index: bool = False
    ):
        self.logger = logger
        self.score_function = score_function  # "cos_sim" or "dot_product"
        self.save_index = save_index
        self.index: Optional[faiss.Index] = None

    def load(self, path: str) -> Optional[faiss.Index]:
        try:
            self.index = faiss.read_index(path)
            self.logger.info(f"FAISS index loaded from {path}")
        except Exception as e:
            self.logger.warning(f"Could not load FAISS index from {path}: {e}")
            return None

    def create_index(self, dimension: int) -> faiss.Index:
        # For dot_product we still use inner product metric; for cosine we normalize vectors.
        M = 32
        self.index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
        # Reasonable defaults; can be tuned per workload
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 64
        self.logger.info(
            f"Created HNSWFlat index (dim={dimension}, M={M}, metric=IP)."
        )

    def add(self, embeddings: np.ndarray):
        assert self.index is not None, "Index is not created"
        # Cosine similarity -> normalize vectors BEFORE adding
        if self.score_function == "cos_sim":
            faiss.normalize_L2(embeddings)
        self.index.add(np.ascontiguousarray(embeddings.astype(np.float32)))
        self.logger.info(f"Added {embeddings.shape[0]} vectors to FAISS index.")

    def search(self, query_embeddings: np.ndarray, k: int):
        assert self.index is not None, "Index is not created/loaded"
        # Cosine similarity -> normalize queries at search time
        qe = np.ascontiguousarray(query_embeddings.astype(np.float32))
        if self.score_function == "cos_sim":
            faiss.normalize_L2(qe)

        return self.index.search(qe, k)

    def retrieve(
            self,
            query_embeddings: np.ndarray,
            query_ids: List[str],
            corpus: Any, k: int
    ) -> List[Dict[str, Dict[str, float]]]:
        results = {q_id: {} for q_id in query_ids}

        if self.score_function == "cos_sim":
            self.logger.info("normalizing embeddings")
            faiss.normalize_L2(query_embeddings)

        scores, indices = self.index.search(query_embeddings.astype(np.float32), k)

        for idx, query_id in enumerate(query_ids):
            for corpus_idx, score in zip(indices[idx], scores[idx]):
                corpus_id = corpus[int(corpus_idx)]["concept_id"]
                results[query_id][corpus_id] = float(score)

        final_results = []
        for k, v in results.items():
            final_results.append({k: v})
        return final_results
