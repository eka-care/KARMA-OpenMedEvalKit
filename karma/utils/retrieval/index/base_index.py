from abc import ABC, abstractmethod
import numpy as np

class BaseIndex(ABC):
    @abstractmethod
    def retrieve(self, embeddings: np.ndarray, ids: np.ndarray | None = None):
        """Add embeddings to the index, optionally with custom IDs."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load the index from disk."""
        pass
