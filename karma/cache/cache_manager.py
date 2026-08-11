import base64
import hashlib
import json
import multiprocessing
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import logging
from karma.cache.duckdb_cache_io import DuckDBCacheIO
from karma.cache.dynamodb_cache_io import DynamoDBCacheIO
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.data_models.model_meta import ModelMeta
from karma.utils.git import get_current_commit_hash

logger = logging.getLogger(__name__)

_MD5_HEX = set("0123456789abcdef")


def _config_hash_from_env() -> Optional[str]:
    """Return normalized config hash from KARMA_CONFIG_HASH if valid (32-char MD5 hex)."""

    raw = os.environ.get("KARMA_CONFIG_HASH", "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    if len(lowered) != 32 or any(ch not in _MD5_HEX for ch in lowered):
        logger.warning(
            "KARMA_CONFIG_HASH is set but ignored (expected 32 hex characters): %r",
            raw,
        )
        return None
    return lowered


_forced_job_id_claimed = False

PROCESS_JOB_IDS: list = []


def _claim_forced_job_id() -> Optional[str]:
    """KARMA_JOB_ID applies to the first run in a process only (one job = one dataset)."""

    global _forced_job_id_claimed
    forced = os.environ.get("KARMA_JOB_ID", "").strip() or None
    if not forced or _forced_job_id_claimed:
        return None
    _forced_job_id_claimed = True
    return forced


def _create_cache_io(
    *,
    cache_type: Optional[str] = None,
    cache_path: Optional[str] = None,
    fallback_duckdb_path: str = "cache/benchmark_cache.duckdb",
    read_only: bool = False,
):
    """Instantiate a cache IO backend based on configuration."""

    resolved_cache_type = (cache_type or os.getenv("KARMA_CACHE_TYPE", "duckdb")).lower()
    resolved_cache_path = cache_path if cache_path is not None else os.getenv(
        "KARMA_CACHE_PATH", fallback_duckdb_path
    )

    if resolved_cache_type == "duckdb":
        db_path = Path(str(resolved_cache_path)).expanduser()
        return DuckDBCacheIO(db_path=str(db_path), read_only=read_only)
    if resolved_cache_type == "dynamodb":
        return DynamoDBCacheIO()
    raise ValueError(
        f"Unsupported cache type: {resolved_cache_type}. "
        "Supported types: duckdb, dynamodb"
    )


def create_cache_io(
    *,
    cache_type: Optional[str] = None,
    cache_path: Optional[str] = None,
    fallback_duckdb_path: str = "cache/benchmark_cache.duckdb",
    read_only: bool = False,
):
    """Public helper to instantiate a cache IO backend."""

    return _create_cache_io(
        cache_type=cache_type,
        cache_path=cache_path,
        fallback_duckdb_path=fallback_duckdb_path,
        read_only=read_only,
    )


def export_run_samples_to_parquet(
    *,
    config_hash: str,
    file_path: str | Path,
    cache_path: Optional[str] = None,
    cache_type: Optional[str] = None,
) -> Tuple[str, int]:
    """Export all cached samples for a run to a Parquet file."""

    cache_io = _create_cache_io(
        cache_type=cache_type,
        cache_path=cache_path,
        fallback_duckdb_path="cache/benchmark_cache.duckdb",
        read_only=True,
    )
    try:
        exporter = getattr(cache_io, "export_run_samples_to_parquet", None)
        if callable(exporter):
            return exporter(
                config_hash=config_hash,
                file_path=file_path,
            )
        raise ValueError(
            f"Cache backend {type(cache_io).__name__} does not support Parquet export"
        )
    finally:
        close_method = getattr(cache_io, "close_connections", None)
        if callable(close_method):
            close_method()


class CacheManager:
    """
    Optimized cache manager with centralized cache operations and reduced redundancy.

    Handles hash generation and cache operations with configurable backends.
    """

    def __init__(
        self,
        dataset_name: str,
        model_config: Any,
        run_instance_id: Optional[str] = None,
        *,
        dataset_subset: Optional[str] = None,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.model_config = model_config
        self.run_instance_id = run_instance_id
        self.max_samples = max_samples
        from karma.utils.git import get_current_commit_hash
        self.git_commit_hash = get_current_commit_hash(Path(__file__))
        self._config_payload = self._build_config_payload(model_config)
        self._model_hash_payload = self.build_model_payload(model_config)
        self.dataset_subset = dataset_subset
        self.split = split
        self._run_metadata: Dict[str, Any] = {}

        self.database_hits = 0
        self.database_misses = 0

        self.cache_io = _create_cache_io(read_only=False)
        self.config_hash = self.initialize_run(
            model_config, dataset_name, dataset_subset, split
        )

    @staticmethod
    def _merge_runner_payload_into_config(
        model_config_dict: Dict[str, Any],
        existing_metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Preserve draft `_runner_payload` on the run row when rebinding an existing config_hash."""

        if not existing_metadata:
            return model_config_dict
        raw_mc = existing_metadata.get("model_config")
        if raw_mc is None:
            return model_config_dict
        try:
            prior = json.loads(raw_mc) if isinstance(raw_mc, str) else raw_mc
        except json.JSONDecodeError:
            return model_config_dict
        if not isinstance(prior, dict):
            return model_config_dict
        runner = prior.get("_runner_payload")
        if runner is None:
            return model_config_dict
        merged = dict(model_config_dict)
        merged["_runner_payload"] = runner
        return merged

    def _generate_cache_key(self, model_input: Dict[str, Any]) -> Tuple[str, str]:
        """
        Centralized cache key generation to eliminate redundancy.

        Args:
            model_input: The input/prompt to generate cache key for

        Returns:
            Tuple of (input_hash, cache_key)
        """
        # Generate input hash over the generation-input whitelist only
        input_hash = self.generate_hash(self.build_sample_payload(model_input))

        # Generate cache key with the model hash payload (loader_kwargs only)
        cache_key_data = {**self._model_hash_payload, "input_hash": input_hash}
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

    def get_cache_identifiers_for_sample(
        self, sample: DataLoaderIterable | Dict[str, Any]
    ) -> Dict[str, str]:
        """Return cache key and row hash for a serialized sample."""

        if isinstance(sample, DataLoaderIterable):
            sample_payload = sample.model_dump()
        else:
            sample_payload = sample

        input_hash, cache_key = self._generate_cache_key(sample_payload)
        return {
            "dataset_row_hash": input_hash,
            "cache_key": cache_key,
        }

    # Keys that contain audio or image binary data and should be stripped from cache storage
    _MEDIA_KEYS = frozenset({
        # Audio-related keys
        "audio",
        "audio_array",
        "audio_bytes",
        "audio_data",
        "waveform",
        "audio_input",
        "audio_sample",
        "raw_audio",
        # Image-related keys
        "image",
        "images",
        "image_bytes",
        "image_data",
        "pixel_values",
        "image_input",
        "raw_image",
    })

    @staticmethod
    def _strip_media_bytes(obj: Any) -> Any:
        """
        Remove audio and image binary data from objects before cache storage.

        This prevents large binary payloads from bloating the cache.

        Args:
            obj: Object to strip media bytes from

        Returns:
            Object with media bytes removed (replaced with placeholder strings)
        """
        # Handle numpy arrays - these are typically audio waveforms
        if np is not None and isinstance(obj, np.ndarray):
            return f"<numpy.ndarray shape={obj.shape} dtype={obj.dtype}>"

        # Handle PIL Images
        if Image is not None and isinstance(obj, Image.Image):
            return f"<PIL.Image mode={obj.mode} size={obj.size}>"

        # Handle bytes objects
        if isinstance(obj, bytes):
            return f"<bytes len={len(obj)}>"

        # Handle dictionaries recursively, removing known media keys
        if isinstance(obj, dict):
            return {
                k: (
                    f"<{k} stripped>"
                    if k.lower() in CacheManager._MEDIA_KEYS
                    else CacheManager._strip_media_bytes(v)
                )
                for k, v in obj.items()
            }

        # Handle lists recursively
        if isinstance(obj, (list, tuple)):
            return [CacheManager._strip_media_bytes(item) for item in obj]

        # Return as-is for basic types (str, int, float, bool, None)
        return obj

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

    # Infra keys forbidden in loader_kwargs (models spec M3) — validated, never stripped.
    _HASH_DENYLIST = frozenset(
        {
            "device",
            "timeout",
            "max_retries",
            "max_workers",
            "api_key",
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "region_name",
            "region",
            "s3_bucket",
            "s3_key_prefix",
            "server_startup_timeout",
            "sleep_duration",
        }
    )

    # Generation-input whitelist for cache keys (caching spec C2).
    _SAMPLE_HASH_KEYS = ("input", "conversation", "system_prompt", "images", "audio", "language", "tool_policy")

    @staticmethod
    def build_model_payload(model_config: ModelMeta) -> Dict[str, Any]:
        """Model half of every hash: loader_kwargs + injected model_name_or_path.
        revision and loader_class are deliberately not hashed (specs M5/M6)."""

        kwargs = dict(model_config.loader_kwargs or {})
        offending = CacheManager._HASH_DENYLIST & set(kwargs)
        if offending:
            raise ValueError(
                "Infra keys are not allowed in loader_kwargs "
                f"{sorted(offending)} — move them to __init__ defaults "
                f"(model '{model_config.name}'; for a device override set "
                "the KARMA_DEVICE environment variable)."
            )
        kwargs["model_name_or_path"] = model_config.model_path or model_config.name
        return kwargs

    @staticmethod
    def build_sample_payload(sample: Any) -> Dict[str, Any]:
        """Sample half of a cache_key: generation inputs only (typed root fields;
        other_args is metric metadata and never hashed)."""

        if isinstance(sample, DataLoaderIterable):
            data = sample.model_dump()
        else:
            data = dict(sample)

        payload: Dict[str, Any] = {}
        for key in CacheManager._SAMPLE_HASH_KEYS:
            value = data.get(key)
            if value is not None:
                payload[key] = value

        return payload

    @staticmethod
    def config_data_for_run_hash(
        model_config: ModelMeta,
        dataset_name: str,
        *,
        dataset_subset: Optional[str],
        dataset_split: Optional[str],
        run_instance_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the dict passed to :meth:`generate_hash` in :meth:`initialize_run`
        (single source of truth). max_samples is a job column, never hashed (C4)."""

        payload = CacheManager.build_model_payload(model_config)
        if run_instance_id:
            payload["_run_instance_id"] = run_instance_id
        return {
            **payload,
            "dataset_name": dataset_name,
            "dataset_subset": dataset_subset,
            "dataset_split": dataset_split,
        }

    @staticmethod
    def hash_for_run_config(
        model_config: ModelMeta,
        dataset_name: str,
        *,
        dataset_subset: Optional[str] = None,
        dataset_split: Optional[str] = None,
        run_instance_id: Optional[str] = None,
    ) -> str:
        """MD5 config hash for a run — matches :meth:`initialize_run` when ``KARMA_CONFIG_HASH`` is unset."""

        data = CacheManager.config_data_for_run_hash(
            model_config,
            dataset_name,
            dataset_subset=dataset_subset,
            dataset_split=dataset_split,
            run_instance_id=run_instance_id,
        )
        return CacheManager.generate_hash(data)

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
                json_str = json.dumps(
                    serializable_data, sort_keys=True, separators=(",", ":")
                )
            else:
                json_str = str(serializable_data)

            # Use MD5 for fast hashing (cache keys don't need cryptographic security)
            return hashlib.md5(json_str.encode()).hexdigest()

        except Exception:
            # Fallback: convert to string and hash directly
            fallback_str = str(data)
            return hashlib.md5(fallback_str.encode()).hexdigest()

    @staticmethod
    def _build_metric_sort_key(
        *,
        metric_name: str,
        target_scope: str,
        target_id: str,
    ) -> str:
        """Compose a Dynamo-friendly sort key that preserves uniqueness semantics."""

        scope = target_scope or "run"
        identifier = target_id or "__run__"
        return f"{scope}#{identifier}#{metric_name}"

    def initialize_run(
        self,
        model_config: ModelMeta,
        dataset_name: str,
        dataset_subset: Optional[str] = None,
        dataset_split: Optional[str] = None,
    ) -> str:
        """
        Initialize a new run and return config hash.

        Args:
            model_config: Model configuration
            dataset_name: Dataset registry identifier or path
            dataset_subset: Optional subset/config value for the dataset
            dataset_split: Optional split value for the dataset

        Returns:
            Configuration hash for this run
        """
        model_config_dict = self._config_payload.copy()
        subset_value = dataset_subset if dataset_subset is not None else self.dataset_subset
        split_value = dataset_split if dataset_split is not None else self.split
        config_data = self.__class__.config_data_for_run_hash(
            model_config,
            dataset_name,
            dataset_subset=subset_value,
            dataset_split=split_value,
            run_instance_id=self.run_instance_id,
        )
        self.job_id = _claim_forced_job_id() or str(uuid.uuid4())
        PROCESS_JOB_IDS.append(self.job_id)
        forced_hash = _config_hash_from_env()
        if forced_hash:
            config_hash = forced_hash
            getter = getattr(self.cache_io, "get_run_metadata", None)
            existing = getter(config_hash) if callable(getter) else None
            model_config_dict = self._merge_runner_payload_into_config(
                model_config_dict, existing
            )
            logger.info(
                "Using KARMA_CONFIG_HASH=%s for this run (skipping derived config hash)",
                config_hash,
            )
        else:
            config_hash = self.generate_hash(config_data)

        # Store run metadata
        run_data = {
            "config_hash": config_hash,
            "status": "running",
            "model_config": json.dumps(model_config_dict),
            "model_id": model_config.name,
            "model_path": model_config.model_path,
            "dataset_name": dataset_name,
            "dataset_subset": subset_value,
            "split": split_value,
        }
        
        run_data["job_id"] = self.job_id

        if self.git_commit_hash:
            run_data["git_commit_hash"] = self.git_commit_hash

        if self.max_samples is not None:
            run_data["max_samples"] = self.max_samples
        from karma.utils.run_mode import derive_run_mode

        run_data["run_mode"] = derive_run_mode(max_samples=self.max_samples)

        self.cache_io.save_run_metadata(run_data)
        self._run_metadata = run_data

        # Cache current config for this session
        self.config_hash = config_hash
        self.model_config = model_config
        self.dataset_name = dataset_name

        return config_hash

    def batch_fetch_rows(
        self, model_inputs: List[DataLoaderIterable]
    ) -> List[Dict[str, Any] | None]:
        """
        Fetch multiple cached inference results.

        Args:
            model_inputs: List of model inputs/prompts

        Returns:
            Dictionary mapping model inputs to cached results (None if not found)
        """
        model_inputs = [
            input.model_dump() if not isinstance(input, dict) else input
            for input in model_inputs
        ]

        # Use centralized batch cache key generation
        _, cache_keys = self._batch_generate_cache_keys(model_inputs)
        # Batch fetch from database
        cache_results = self.cache_io.batch_get_inference_results(cache_keys)
        # Read-only fetch (spec C7); failed rows are misses (C8) unless raw
        # output exists for re-extraction (C10 — parser fix must not re-bill).
        result: List[Dict[str, Any] | None] = []
        hit_keys: List[str] = []
        gt_updates: List[tuple] = []
        for i, cache_key in enumerate(cache_keys):
            cached_row = cache_results.get(cache_key)
            if cached_row is not None and not cached_row.get("success", True):
                # NULL raw = legacy row; "" = stored blank response, reusable.
                reusable = (
                    cached_row.get("raw_output") is not None
                    and cached_row.get("error_type") != "model_error"
                )
                if not reusable:
                    cached_row = None
            if cached_row is not None:
                hit_keys.append(cache_key)
                new_gt = model_inputs[i].get("expected_output")
                old_gt = cached_row.get("ground_truth_output")
                if (new_gt if new_gt is not None else "") != (
                    old_gt if old_gt is not None else ""
                ):
                    gt_updates.append((cache_key, new_gt))
                    cached_row["ground_truth_output"] = new_gt
            result.append(cached_row)

        self.database_hits += len(hit_keys)
        self.database_misses += len(result) - len(hit_keys)
        self._link_run_samples(hit_keys)
        self._refresh_ground_truth(gt_updates)
        return result

    def batch_save_rows(self, batch_data: List[Dict[str, Any]], dataset_name) -> bool:
        """
        Save multiple inference results to database cache.

        Args:
            batch_data: List of dictionaries containing inference data with keys:
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
            if data.get("error_type") == "model_error":
                continue
            sample = data["sample"]
            # Stored model_input = the hashed generation inputs, media stripped.
            stripped_input = CacheManager._strip_media_bytes(
                CacheManager.build_sample_payload(sample)
            )
            serializable_input = CacheManager._make_serializable(stripped_input)
            model_input_payload = json.dumps(serializable_input)
            inference_data = {
                "cache_key": cache_keys[i],
                "dataset_name": dataset_name,
                "dataset_row_metadata": sample.get("other_args", {}),
                "dataset_row_hash": input_hashes[i],
                "model_input": model_input_payload,
                "model_output": data.get("prediction") or data.get("raw_output", ""),
                "raw_output": data.get("raw_output", ""),
                "processed_output": data.get("prediction"),
                "error_type": data.get("error_type"),
                "ground_truth_output": data.get("expected_output", ""),
                "tool_trace": data.get("tool_trace", ""),
                "config_hash": self.config_hash,
                "success": data.get("success", True),
            }
            inference_data_list.append(inference_data)

        # Save to database
        saved = self.cache_io.batch_save_inference_results(inference_data_list)
        if saved:
            self._link_run_samples([d["cache_key"] for d in inference_data_list])
        return saved

    def _refresh_ground_truth(self, updates: List[tuple]) -> None:
        """Sync stored ground truth to the dataset's current expected output."""

        if not updates:
            return
        updater = getattr(self.cache_io, "batch_update_ground_truth", None)
        if not callable(updater):
            return
        try:
            updater(updates)
        except Exception:
            logger.exception(
                "Failed to refresh ground truth for %d cached rows", len(updates)
            )

    def _link_run_samples(self, cache_keys: List[str]) -> None:
        """Record per-job sample membership in run_samples."""

        if not cache_keys:
            return
        job_id = getattr(self, "job_id", None)
        linker = getattr(self.cache_io, "batch_link_run_samples", None)
        if not job_id or not callable(linker):
            return
        try:
            linker(job_id, cache_keys)
        except Exception:
            logger.exception(
                "Failed to link %d samples to job %s", len(cache_keys), job_id
            )

    def batch_save_rubric_question_results(
        self, records: List[Dict[str, Any]]
    ) -> bool:
        """Persist rubric evaluation payloads for downstream analysis."""

        return self.cache_io.batch_save_rubric_question_results(records)

    def save_rubric_outputs(self, records: List[Dict[str, Any]]) -> bool:
        """Persist content-addressed judge outputs (rubric_outputs table)."""

        saver = getattr(self.cache_io, "save_rubric_outputs", None)
        if saver is None:
            return False
        return saver(records)

    def get_rubric_outputs(
        self, judge_hashes: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch stored judge outputs by judge_hash for read-back."""

        getter = getattr(self.cache_io, "get_rubric_outputs", None)
        if getter is None:
            return {}
        return getter(judge_hashes)

    def save_metric_results(self, records: List[Dict[str, Any]]) -> bool:
        """Persist generic metric outputs for runs and samples."""

        if not records:
            return True

        serialized: List[Dict[str, Any]] = []
        for record in records:
            metric_name = str(record["metric_name"])
            target_scope = str(record.get("target_scope", "run"))
            target_id = str(record.get("target_id", "__run__"))

            payload = record.get("payload")
            payload_json: Optional[str] = None
            if payload is not None:
                serializable_payload = CacheManager._make_serializable(payload)
                payload_json = json.dumps(serializable_payload)

            value = record.get("value")
            numeric_value: Optional[float]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_value = float(value)
            else:
                numeric_value = None

            serialized.append(
                {
                    "job_id": getattr(self, "job_id", None),
                    "config_hash": self.config_hash,
                    "dataset_name": self.dataset_name,
                    "metric_name": metric_name,
                    "target_scope": target_scope,
                    "target_id": target_id,
                    "target_sort_key": CacheManager._build_metric_sort_key(
                        metric_name=metric_name,
                        target_scope=target_scope,
                        target_id=target_id,
                    ),
                    "cache_key": record.get("cache_key"),
                    "value": numeric_value,
                    "payload_json": payload_json,
                    "total_count": int(record["total_count"])
                    if record.get("total_count") is not None
                    else None,
                    "scored_count": int(record["scored_count"])
                    if record.get("scored_count") is not None
                    else None,
                }
            )

        return self.cache_io.batch_save_metric_results(serialized)

    def invalidate_cache_for_keys(self, cache_keys: List[str]) -> bool:
        """
        Invalidate specific cache entries by keys.

        Args:
            cache_keys: List of cache keys to invalidate

        Returns:
            True if successfully invalidated, False otherwise
        """
        try:
            # This would need to be implemented in the cache_io backends
            # For now, we can implement a simple version
            return self.cache_io.invalidate_cache_keys(cache_keys)
        except AttributeError:
            # Fallback if cache_io doesn't support invalidation
            return False

    def close_connections(self):
        """Close cache connections."""
        self.cache_io.close_connections()

    def record_dataset_metadata(
        self, *, dataset_subset: Optional[str] = None, split: Optional[str] = None
    ) -> None:
        """Persist dataset-specific metadata for this run when available."""

        if not dataset_subset and not split:
            return

        if dataset_subset:
            self.dataset_subset = dataset_subset
            self._run_metadata["dataset_subset"] = dataset_subset
        if split:
            self.split = split
            self._run_metadata["split"] = split

        if self._run_metadata:
            self.cache_io.save_run_metadata(self._run_metadata)

    def _build_config_payload(self, model_config: ModelMeta) -> Dict[str, Any]:
        payload = model_config.model_dump(exclude_none=True)
        if self.run_instance_id:
            payload["_run_instance_id"] = self.run_instance_id
        return payload


def export_run_to_dynamodb(
    *,
    config_hash: str,
    cache_path: Optional[str] = None,
    cache_type: Optional[str] = None,
) -> Dict[str, int]:
    """Export all cached data for a run from a local backend (DuckDB) to DynamoDB.

    Args:
        config_hash: The configuration hash identifying the run to export.
        cache_path: Path to the local cache file.
        cache_type: Type of the source cache backend (defaults to env/duckdb).

    Returns:
        Dictionary with counts of exported items per table.
    """
    from karma.cache.constants import RUN_METADATA_TABLE

    source = _create_cache_io(
        cache_type=cache_type,
        cache_path=cache_path,
        fallback_duckdb_path="./cache.db",
        read_only=True,
    )

    if not isinstance(source, DuckDBCacheIO):
        raise ValueError("Source cache must be a DuckDB backend")

    dest = DynamoDBCacheIO()

    try:
        # --- 1. Run metadata ---
        row = source.db_io.fetchone(
            f"SELECT config_hash, model_config, model_id, dataset_name, "
            f"dataset_subset, split, git_commit_hash, task, remarks "
            f"FROM {RUN_METADATA_TABLE} WHERE config_hash = ? "
            f"ORDER BY created_at DESC LIMIT 1",
            [config_hash],
        )
        if row is None:
            raise ValueError(f"Run with config_hash={config_hash} not found")

        run_data = {
            "config_hash": row[0],
            "model_config": row[1],
            "model_id": row[2],
            "dataset_name": row[3],
            "dataset_subset": row[4],
            "split": row[5],
            "git_commit_hash": row[6],
            "task": row[7],
            "remarks": row[8],
        }
        dest.save_run_metadata(run_data)
        logger.info("Exported run metadata for %s", config_hash)

        # --- 2. Inference results ---
        inference_rows = source.db_io.fetchall(
            "SELECT cache_key, dataset_name, dataset_row_metadata, dataset_row_hash, "
            "model_input, model_output, ground_truth_output, config_hash, success "
            "FROM inference_results WHERE config_hash = ?",
            [config_hash],
        )

        inference_items: List[Dict[str, Any]] = []
        for r in inference_rows:
            metadata = r[2]
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    pass

            model_input = r[4]
            if isinstance(model_input, str):
                try:
                    model_input = json.loads(model_input)
                except (json.JSONDecodeError, TypeError):
                    pass

            inference_items.append({
                "cache_key": r[0],
                "dataset_name": r[1],
                "dataset_row_metadata": metadata,
                "dataset_row_hash": r[3],
                "model_input": model_input,
                "model_output": r[5],
                "ground_truth_output": r[6],
                "config_hash": r[7],
                "success": bool(r[8]),
            })

        if inference_items:
            dest.batch_save_inference_results(inference_items)
        logger.info("Exported %d inference results", len(inference_items))

        # --- 3. Rubric evaluation results ---
        rubric_count = 0
        try:
            rubric_rows = source.db_io.fetchall(
                "SELECT cache_key, dataset_name, config_hash, dataset_row_hash, "
                "metric_name, question_index, question_score, "
                "rubric_evaluations_json, metric_metadata_json "
                "FROM rubric_evaluation_results WHERE config_hash = ?",
                [config_hash],
            )

            rubric_items: List[Dict[str, Any]] = []
            for r in rubric_rows:
                rubric_evals = r[7]
                if isinstance(rubric_evals, str):
                    try:
                        rubric_evals = json.loads(rubric_evals)
                    except (json.JSONDecodeError, TypeError):
                        pass

                metric_meta = r[8]
                if isinstance(metric_meta, str):
                    try:
                        metric_meta = json.loads(metric_meta)
                    except (json.JSONDecodeError, TypeError):
                        pass

                rubric_items.append({
                    "cache_key": r[0],
                    "dataset_name": r[1],
                    "config_hash": r[2],
                    "dataset_row_hash": r[3],
                    "metric_name": r[4],
                    "question_index": int(r[5]),
                    "question_score": float(r[6]) if r[6] is not None else None,
                    "rubric_evaluations": rubric_evals,
                    "metric_metadata": metric_meta,
                })

            if rubric_items:
                dest.batch_save_rubric_question_results(rubric_items)
            rubric_count = len(rubric_items)
            logger.info("Exported %d rubric evaluation results", rubric_count)
        except Exception:
            logger.info("No rubric evaluation results to export (table may not exist)")

        # --- 4. Metric results ---
        metric_count = 0
        try:
            metric_rows = source.db_io.fetchall(
                "SELECT config_hash, metric_name, target_scope, target_id, "
                "value, payload_json, total_count, scored_count "
                "FROM metric_results WHERE config_hash = ?",
                [config_hash],
            )

            metric_items: List[Dict[str, Any]] = []
            for r in metric_rows:
                metric_name = r[1]
                target_scope = r[2] or "run"
                target_id = r[3] or "__run__"

                metric_items.append({
                    "config_hash": r[0],
                    "dataset_name": run_data.get("dataset_name"),
                    "metric_name": metric_name,
                    "target_scope": target_scope,
                    "target_id": target_id,
                    "target_sort_key": CacheManager._build_metric_sort_key(
                        metric_name=metric_name,
                        target_scope=target_scope,
                        target_id=target_id,
                    ),
                    "cache_key": None,
                    "value": float(r[4]) if r[4] is not None else None,
                    "payload_json": r[5],
                    "total_count": int(r[6]) if r[6] is not None else None,
                    "scored_count": int(r[7]) if r[7] is not None else None,
                })

            if metric_items:
                dest.batch_save_metric_results(metric_items)
            metric_count = len(metric_items)
            logger.info("Exported %d metric results", metric_count)
        except Exception:
            logger.info("No metric results to export (table may not exist)")

        return {
            "run_metadata": 1,
            "inference_results": len(inference_items),
            "rubric_results": rubric_count,
            "metric_results": metric_count,
        }
    finally:
        source.close_connections()
        dest.close_connections()
