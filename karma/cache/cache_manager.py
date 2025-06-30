import hashlib
import json
import base64
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

from typing import Any, Dict, List, Tuple
from karma.models.base import ModelConfig
from karma.cache.duckdb_cache_io import DuckDBCacheIO
from karma.cache.dynamodb_cache_io import DynamoDBCacheIO


class CacheManager:
    """
    Optimized cache manager with centralized cache operations and reduced redundancy.

    Handles hash generation and cache operations with configurable backends.
    """

    def __init__(self, dataset_name: str, model_config: ModelConfig):
        self.model_config = model_config

        self.database_hits = 0
        self.database_misses = 0

        # Get cache configuration from environment variables
        cache_type = os.getenv("KARMA_CACHE_TYPE", "duckdb").lower()
        cache_path = os.getenv("KARMA_CACHE_PATH", "./cache.db")

        # Initialize cache based on type
        if cache_type == "duckdb":
            self.cache_io = DuckDBCacheIO(db_path=cache_path)
        elif cache_type == "dynamodb":
            self.cache_io = DynamoDBCacheIO()
        else:
            raise ValueError(
                f"Unsupported cache type: {cache_type}. Supported types: duckdb, dynamodb"
            )

        self.initialize_run(model_config, dataset_name)

    def format_model_input(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        processed_model_input = {
            "prompt": model_input.get("input", ""),
            "images": model_input.get("images", []),
            "audios": model_input.get("audios", []),
        }
        return processed_model_input

    def _generate_cache_key(self, model_input: Dict[str, Any]) -> Tuple[str, str]:
        """
        Centralized cache key generation to eliminate redundancy.

        Args:
            model_input: The input/prompt to generate cache key for

        Returns:
            Tuple of (input_hash, cache_key)
        """
        # Generate input hash
        input_hash = self.generate_hash(self.format_model_input(model_input))

        # Generate cache key with model config
        cache_key_data = {**self.model_config.to_dict(), "input_hash": input_hash}
        cache_key = self.generate_hash(cache_key_data)

        return input_hash, cache_key

    def _batch_generate_cache_keys(
        self, model_inputs: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """
        Batch cache key generation for multiple inputs.

        Args:
            model_inputs: List of model inputs/prompts

        Returns:
            Tuple of (input_hashes, cache_keys)
        """
        # Use ThreadPoolExecutor for CPU-bound hash generation
        # executor.map() preserves order - results will be in same order as inputs
        max_workers = min(len(model_inputs), multiprocessing.cpu_count())

        # Prepare data for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self._generate_cache_key, model_inputs))

        # Unpack results - order is preserved from input order
        input_hashes = [result[0] for result in results]
        cache_keys = [result[1] for result in results]

        return input_hashes, cache_keys

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """
        Convert non-JSON-serializable objects to their byte representations.

        Args:
            obj: Object to make serializable

        Returns:
            Serializable representation of the object
        """

        # Handle numpy arrays
        if np is not None and isinstance(obj, np.ndarray):
            return base64.b64encode(obj.tobytes()).decode("utf-8")

        # Handle PIL Images
        if Image is not None and isinstance(obj, Image.Image):
            import io

            img_bytes = io.BytesIO()
            obj.save(img_bytes, format="PNG")
            return base64.b64encode(img_bytes.getvalue()).decode("utf-8")

        # Handle bytes objects
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode("utf-8")

        # Handle dictionaries recursively
        if isinstance(obj, dict):
            return {k: CacheManager._make_serializable(v) for k, v in obj.items()}

        # Handle lists recursively
        if isinstance(obj, (list, tuple)):
            return [CacheManager._make_serializable(item) for item in obj]

        # Return as-is for basic types (str, int, float, bool, None)
        return obj

    @staticmethod
    def generate_hash(data: Any) -> str:
        """
        Generate hash from data using MD5 for cache keys.

        Args:
            data: Data to hash (will be made JSON serializable)

        Returns:
            Hex digest of the hash
        """
        try:
            # For strings (like prompts), hash directly for speed
            if isinstance(data, str):
                return hashlib.md5(data.encode()).hexdigest()

            # For objects, use JSON serialization
            serializable_data = CacheManager._make_serializable(data)

            if isinstance(serializable_data, (dict, list)):
                json_str = json.dumps(serializable_data)
            else:
                json_str = str(serializable_data)

            # Use MD5 for fast hashing (cache keys don't need cryptographic security)
            return hashlib.md5(json_str.encode()).hexdigest()

        except Exception:
            # Fallback: convert to string and hash directly
            fallback_str = str(data)
            return hashlib.md5(fallback_str.encode()).hexdigest()

    def initialize_run(self, model_config: ModelConfig, dataset_name: str) -> str:
        """
        Initialize a new run and return config hash.

        Args:
            model_config: Model configuration
            dataset_name: Dataset name (HuggingFace path)

        Returns:
            Configuration hash for this run
        """
        # Generate config hash directly
        config_data = {**model_config.to_dict(), "dataset_name": dataset_name}
        config_hash = self.generate_hash(config_data)

        # Store run metadata
        run_data = {
            "config_hash": config_hash,
            "model_config": json.dumps(model_config.to_dict()),
            "model_id": model_config.model_id,
            "dataset_name": dataset_name,
        }

        self.cache_io.save_run_metadata(run_data)

        # Cache current config for this session
        self.config_hash = config_hash
        self.model_config = model_config

        return config_hash

    def batch_fetch_rows(
        self, model_inputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any] | None]:
        """
        Fetch multiple cached inference results.

        Args:
            model_inputs: List of model inputs/prompts

        Returns:
            Dictionary mapping model inputs to cached results (None if not found)
        """

        # Use centralized batch cache key generation
        _, cache_keys = self._batch_generate_cache_keys(model_inputs)
        # Batch fetch from database
        cache_results = self.cache_io.batch_get_inference_results(cache_keys)
        # Process results and update statistics
        result = [cache_results.get(cache_key, None) for cache_key in cache_keys]

        hits = len([x for x in result if x is not None])
        self.database_hits += hits
        self.database_misses += len(result) - hits
        return result

    def batch_save_rows(self, batch_data: List[Dict[str, Any]]) -> bool:
        """
        Save multiple inference results to database cache.

        Args:
            batch_data: List of dictionaries containing inference data with keys:
                       - model_input: The actual input/prompt sent to the model
                       - model_output: Model's output
                       - model_output_reasoning: Model's reasoning (optional)
                       - ground_truth_output: Ground truth output (optional)
                       - ground_truth_reasoning: Ground truth reasoning (optional)

        Returns:
            True if successfully cached, False otherwise
        """
        # Prepare batch data for database
        inference_data_list = []
        model_inputs = [data["sample"] for data in batch_data]
        # Use centralized batch cache key generation
        input_hashes, cache_keys = self._batch_generate_cache_keys(model_inputs)

        for i, data in enumerate(batch_data):
            inference_data = {
                "cache_key": cache_keys[i],
                "dataset_row_idx": data["sample"].get("idx", None),
                "dataset_row_metadata": data["sample"].get("meta", {}),
                "dataset_row_hash": input_hashes[i],
                "model_output": data.get("prediction", ""),
                "model_output_reasoning": data.get("thinking_content", ""),
                "ground_truth_output": data.get("expected_output", ""),
                "ground_truth_reasoning": data.get("ground_truth_reasoning", ""),
                "config_hash": self.config_hash,
                "success": data.get("success", True),
            }
            if inference_data.get("model_output"):
                inference_data_list.append(inference_data)

        # Save to database
        return self.cache_io.batch_save_inference_results(inference_data_list)

    def close_connections(self):
        """Close cache connections."""
        self.cache_io.close_connections()
