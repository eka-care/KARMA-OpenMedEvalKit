import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from karma.cache.constants import RUN_METADATA_TABLE
from karma.io.duckdb_io import DuckDBIO


logger = logging.getLogger(__name__)

# Job-keyed run metadata schema for local runs — infra-only fields
# (wandb_run_id, instance_type, k8s meta) deliberately absent.
_RUN_METADATA_DDL_COLUMNS = """
                job_id VARCHAR(36) PRIMARY KEY,
                config_hash VARCHAR(64),
                model_config TEXT,
                model_id VARCHAR(255),
                dataset_name VARCHAR(255),
                dataset_subset VARCHAR(255),
                split VARCHAR(64),
                git_commit_hash VARCHAR(64),
                task TEXT,
                remarks VARCHAR(500),
                status VARCHAR(32),
                run_name VARCHAR(255),
                project_tags TEXT,
                max_samples INTEGER,
                run_mode VARCHAR(8),
                total_samples INTEGER,
                processed_samples INTEGER,
                success_count INTEGER,
                error_count INTEGER,
                cache_hits INTEGER,
                duration_seconds DOUBLE,
                post_processing_error_count INTEGER,
                extras TEXT,
                error_message VARCHAR(1000),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
"""

_METRIC_RESULTS_DDL = """
            CREATE TABLE IF NOT EXISTS metric_results (
                job_id VARCHAR(36) NOT NULL,
                config_hash VARCHAR(64),
                metric_name VARCHAR(128) NOT NULL,
                target_scope VARCHAR(32) NOT NULL,
                target_id VARCHAR(128) NOT NULL,
                value DOUBLE,
                payload_json TEXT,
                total_count BIGINT,
                scored_count BIGINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (job_id, target_scope, target_id, metric_name)
            )
"""


class DuckDBCacheIO:
    """
    Simplified cache IO using DuckDB with direct database queries.

    Architecture:
    - DuckDB: Persistent embedded database storage for all cache data
    - Direct queries: Fast queries by cache_key with indexed lookups
    - Synchronous writes: Direct writes to DuckDB

    Features:
    - Fast indexed reads from DuckDB
    - Direct synchronous writes to DuckDB
    """

    def __init__(
        self,
        db_path: str = "cache/benchmark_cache.duckdb",
        *,
        read_only: bool = False,
    ):
        """Initialize simple cache IO."""
        # Initialize DuckDB (persistent storage) with hardcoded path
        self.read_only = read_only
        self.db_io = DuckDBIO(db_path=db_path, read_only=read_only)
        logger.info("✅ DuckDB persistent storage connected")
        # Initialize schema for benchmark caching when writable
        if not self.read_only:
            self._init_schema()
        self._supports_model_input = self._check_column_exists(
            "inference_results", "model_input"
        )
        self._supports_rubric_table = self._check_table_exists(
            "rubric_evaluation_results"
        )

    def _check_column_exists(self, table: str, column: str) -> bool:
        try:
            rows = self.db_io.fetchall(f"PRAGMA table_info('{table}')")
        except Exception:
            return False
        return any(row[1] == column for row in rows)

    def _check_table_exists(self, table: str) -> bool:
        try:
            row = self.db_io.fetchone(
                """
                SELECT 1
                FROM information_schema.tables
                WHERE lower(table_name) = lower(?)
                LIMIT 1
            """,
                [table],
            )
        except Exception:
            return False
        return row is not None

    def _assert_not_legacy_file(self) -> None:
        """No migration path by decision: pre-v2 local caches must be deleted."""

        if not self._check_table_exists(RUN_METADATA_TABLE):
            return
        row = self.db_io.fetchone(
            """
            SELECT constraint_column_names FROM duckdb_constraints()
            WHERE lower(table_name) = lower(?) AND constraint_type = 'PRIMARY KEY'
        """,
            [RUN_METADATA_TABLE],
        )
        pk = list(row[0]) if row and row[0] else []
        if pk != ["job_id"]:
            raise RuntimeError(
                "This local cache file uses the pre-v2 schema. It is a scratch "
                "cache — delete the file and rerun to recreate it."
            )

    def _init_schema(self):
        """Initialize database schema for benchmark caching."""
        self._assert_not_legacy_file()
        # Create inference results table
        self.db_io.execute("""
            CREATE TABLE IF NOT EXISTS inference_results (
                cache_key VARCHAR(64) PRIMARY KEY,
                dataset_name VARCHAR(64) NOT NULL,
                dataset_row_metadata TEXT,
                dataset_row_hash VARCHAR(64) NOT NULL,
                model_input TEXT,
                model_output TEXT NOT NULL,
                tool_trace TEXT,
                ground_truth_output TEXT,
                config_hash VARCHAR(64) NOT NULL,
                success BOOLEAN NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Run metadata is job-keyed: multiple jobs coexist per config.
        self.db_io.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {RUN_METADATA_TABLE} (
                {_RUN_METADATA_DDL_COLUMNS}
            )
        """
        )
        self.db_io.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_run_metadata_config_created
            ON {RUN_METADATA_TABLE}(config_hash, created_at DESC)
        """
        )

        # Create indices for performance - especially for config_hash lookups
        self.db_io.execute("""
            CREATE INDEX IF NOT EXISTS idx_inference_config_hash 
            ON inference_results(config_hash)
        """)
        self.db_io.execute("""
            CREATE INDEX IF NOT EXISTS idx_inference_cache_key 
            ON inference_results(cache_key)
        """)

        # Store rubric evaluation payloads for error analysis
        self.db_io.execute(
            """
            CREATE TABLE IF NOT EXISTS rubric_evaluation_results (
                cache_key VARCHAR(64) NOT NULL,
                dataset_name VARCHAR(255) NOT NULL,
                config_hash VARCHAR(64) NOT NULL,
                dataset_row_hash VARCHAR(64) NOT NULL,
                metric_name VARCHAR(128) NOT NULL,
                question_index INTEGER NOT NULL,
                question_score DOUBLE,
                rubric_evaluations_json TEXT NOT NULL,
                metric_metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (cache_key, metric_name, question_index)
            )
        """
        )

        # Ensure newer columns exist when upgrading older cache files
        self.db_io.execute(
            """
            ALTER TABLE inference_results
            ADD COLUMN IF NOT EXISTS model_input TEXT
        """
        )
        self.db_io.execute(
            """
            ALTER TABLE inference_results
            ADD COLUMN IF NOT EXISTS tool_trace TEXT
        """
        )

        # Phase 2 columns (change-plan Phase 0 mirror)
        for col, typ in [
            ("raw_output", "TEXT"),
            ("processed_output", "TEXT"),
            ("error_type", "VARCHAR(64)"),
        ]:
            self.db_io.execute(
                f"ALTER TABLE inference_results ADD COLUMN IF NOT EXISTS {col} {typ}"
            )

        # Content-addressed judge cache (rubric_outputs)
        self.db_io.execute(
            """
            CREATE TABLE IF NOT EXISTS rubric_outputs (
                judge_hash VARCHAR(64) PRIMARY KEY,
                cache_key VARCHAR(64) NOT NULL,
                metric_name VARCHAR(128),
                question_index INTEGER,
                judge_model VARCHAR(255),
                judge_model_config TEXT,
                judge_input TEXT,
                question_score DOUBLE,
                criteria TEXT,
                raw_response TEXT,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        self.db_io.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rubric_outputs_cache_metric
            ON rubric_outputs(cache_key, metric_name)
        """
        )

        # Per-job sample membership (run_samples)
        self.db_io.execute(
            """
            CREATE TABLE IF NOT EXISTS run_samples (
                job_id VARCHAR(36) NOT NULL,
                cache_key VARCHAR(64) NOT NULL,
                PRIMARY KEY (job_id, cache_key)
            )
        """
        )
        self.db_io.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_run_samples_cache_key
            ON run_samples(cache_key)
        """
        )

        # Backfill schema for older caches missing the metric metadata column
        self.db_io.execute(
            """
            ALTER TABLE rubric_evaluation_results
            ADD COLUMN IF NOT EXISTS metric_metadata_json TEXT
        """
        )




        self.db_io.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rubric_results_config
            ON rubric_evaluation_results(config_hash)
        """
        )
        self.db_io.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rubric_results_dataset
            ON rubric_evaluation_results(dataset_name)
        """
        )

        self._ensure_metric_results_table()

    def _ensure_metric_results_table(self) -> None:
        """Create the job-keyed metric table."""

        self.db_io.execute(_METRIC_RESULTS_DDL)
        self.db_io.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metric_results_config
            ON metric_results(config_hash)
        """
        )
        self.db_io.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metric_results_metric
            ON metric_results(metric_name)
        """
        )

    def get_inference_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get a cached inference result by cache key from DuckDB."""
        result = self.db_io.fetchone(
            """
            SELECT cache_key, dataset_name, dataset_row_metadata, dataset_row_hash, 
                   model_input, model_output, tool_trace, ground_truth_output, config_hash, success
            FROM inference_results 
            WHERE cache_key = ?
        """,
            [cache_key],
        )

        if result:
            return {
                "cache_key": result[0],
                "dataset_name": result[1],
                "dataset_row_metadata": result[2],
                "dataset_row_hash": result[3],
                "model_input": result[4],
                "model_output": result[5],
                "tool_trace": result[6],
                "ground_truth_output": result[7],
                "config_hash": result[8],
                "success": result[9],
            }
        return None

    def batch_get_inference_results(
        self, cache_keys: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Get multiple cached inference results by cache keys from DuckDB."""
        if not cache_keys:
            return {}

        # Create placeholders for IN clause
        placeholders = ",".join("?" for _ in cache_keys)
        results_data = self.db_io.fetchall(
            f"""
            SELECT cache_key, dataset_name, dataset_row_metadata, dataset_row_hash,
                   model_input, model_output, tool_trace, ground_truth_output, config_hash, success,
                   raw_output, processed_output, error_type
            FROM inference_results
            WHERE cache_key IN ({placeholders})
        """,
            cache_keys,
        )

        results = {}
        for row in results_data:
            results[row[0]] = {
                "cache_key": row[0],
                "dataset_name": row[1],
                "dataset_row_metadata": row[2],
                "dataset_row_hash": row[3],
                "model_input": row[4],
                "model_output": row[5],
                "tool_trace": row[6],
                "ground_truth_output": row[7],
                "config_hash": row[8],
                "success": row[9],
                "raw_output": row[10],
                "processed_output": row[11],
                "error_type": row[12],
            }

        # Add None entries for missing keys
        for cache_key in cache_keys:
            if cache_key not in results:
                results[cache_key] = None

        return results

    def batch_save_inference_results(
        self, inference_data_list: List[Dict[str, Any]]
    ) -> bool:
        """Save multiple inference results to DuckDB."""

        # Handle multiple items
        batch_data = []
        for data in inference_data_list:
            batch_data.append(
                [
                    data["cache_key"],
                    data["dataset_name"],
                    data.get("dataset_row_metadata"),
                    data["dataset_row_hash"],
                    data.get("model_input"),
                    data["model_output"],
                    data.get("tool_trace"),
                    data.get("ground_truth_output"),
                    data["config_hash"],
                    data["success"],
                    data.get("raw_output"),
                    data.get("processed_output"),
                    data.get("error_type"),
                ]
            )

        return self.db_io.executemany(
            """
            INSERT OR REPLACE INTO inference_results
            (cache_key, dataset_name, dataset_row_metadata, dataset_row_hash,
             model_input, model_output, tool_trace, ground_truth_output, config_hash, success,
             raw_output, processed_output, error_type, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            batch_data,
        )

    def get_completed_run_error_count(self, config_hash: str) -> Optional[int]:
        """MAX model errors across completed jobs."""
        if not config_hash:
            return None
        row = self.db_io.fetchone(
            f"SELECT MAX(error_count) FROM {RUN_METADATA_TABLE} "
            "WHERE config_hash = ? AND status = 'completed'",
            [config_hash],
        )
        if not row or row[0] is None:
            return None
        return int(row[0])

    def batch_update_ground_truth(self, updates: List[tuple]) -> bool:
        """Sync stored ground truth to the dataset's current expected output."""
        import json

        if not updates:
            return True
        rows = [
            [
                gt if isinstance(gt, str) or gt is None else json.dumps(gt),
                cache_key,
            ]
            for cache_key, gt in updates
        ]
        return self.db_io.executemany(
            """
            UPDATE inference_results
            SET ground_truth_output = ?, updated_at = CURRENT_TIMESTAMP
            WHERE cache_key = ?
        """,
            rows,
        )

    def batch_link_run_samples(self, job_id: str, cache_keys: List[str]) -> bool:
        """Record per-job sample membership in run_samples."""

        if not job_id or not cache_keys:
            return True
        return self.db_io.executemany(
            """
            INSERT OR IGNORE INTO run_samples (job_id, cache_key)
            VALUES (?, ?)
        """,
            [[job_id, key] for key in cache_keys],
        )

    def batch_save_rubric_question_results(self, records: List[Dict[str, Any]]) -> bool:
        """Persist rubric evaluation records to DuckDB."""

        if not records:
            return True

        batch = []
        for record in records:
            batch.append(
                [
                    record["cache_key"],
                    record["dataset_name"],
                    record["config_hash"],
                    record["dataset_row_hash"],
                    record["metric_name"],
                    record["question_index"],
                    record.get("question_score"),
                    json.dumps(record.get("rubric_evaluations", {})),
                    json.dumps(record.get("metric_metadata", {})),
                ]
            )

        return self.db_io.executemany(
            """
            INSERT OR REPLACE INTO rubric_evaluation_results (
                cache_key,
                dataset_name,
                config_hash,
                dataset_row_hash,
                metric_name,
                question_index,
                question_score,
                rubric_evaluations_json,
                metric_metadata_json,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            batch,
        )

    def save_rubric_outputs(self, records: List[Dict[str, Any]]) -> bool:
        """Upsert judge outputs keyed by content-addressed ``judge_hash``."""

        if not records:
            return True

        batch = []
        for record in records:
            judge_hash = record.get("judge_hash")
            cache_key = record.get("cache_key")
            if not judge_hash or not cache_key:
                continue
            batch.append(
                [
                    judge_hash,
                    cache_key,
                    record.get("metric_name"),
                    record.get("question_index"),
                    record.get("judge_model"),
                    json.dumps(record.get("judge_model_config") or {}),
                    json.dumps(record.get("judge_input") or {}),
                    record.get("question_score"),
                    json.dumps(record.get("criteria") or []),
                    record.get("raw_response"),
                    record.get("error"),
                ]
            )

        if not batch:
            return True

        return self.db_io.executemany(
            """
            INSERT OR REPLACE INTO rubric_outputs (
                judge_hash, cache_key, metric_name, question_index, judge_model,
                judge_model_config, judge_input, question_score, criteria,
                raw_response, error, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            batch,
        )

    def get_rubric_outputs(
        self, judge_hashes: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch stored judge outputs by ``judge_hash``, keyed by that hash."""
        if not judge_hashes:
            return {}

        def _loads(value, default):
            if not value:
                return default
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return default

        placeholders = ",".join(["?"] * len(judge_hashes))
        rows = self.db_io.fetchall(
            f"""
            SELECT judge_hash, cache_key, metric_name, question_index,
                   judge_model, judge_model_config, judge_input, question_score,
                   criteria, raw_response, error
            FROM rubric_outputs WHERE judge_hash IN ({placeholders})
        """,
            list(judge_hashes),
        )

        result: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            result[row[0]] = {
                "judge_hash": row[0],
                "cache_key": row[1],
                "metric_name": row[2],
                "question_index": row[3],
                "judge_model": row[4],
                "judge_model_config": _loads(row[5], {}),
                "judge_input": _loads(row[6], {}),
                "question_score": row[7],
                "criteria": _loads(row[8], []),
                "raw_response": row[9],
                "error": row[10],
            }
        return result

    def batch_save_metric_results(self, records: List[Dict[str, Any]]) -> bool:
        """Persist generic metric outputs (run- or sample-level) to DuckDB."""

        if not records:
            return True

        self._ensure_metric_results_table()

        # Metrics keyed per job; config lookup = legacy fallback (newest job).
        job_id = records[0].get("job_id")
        if not job_id:
            job_id = self._latest_job_for_config(records[0]["config_hash"])
            if not job_id:
                logger.warning(
                    "batch_save_metric_results: no job row for config_hash=%s; "
                    "metric_results not written.",
                    records[0]["config_hash"],
                )
                return False

        batch: List[List[Any]] = []
        for record in records:
            batch.append(
                [
                    str(job_id),
                    record["config_hash"],
                    record["metric_name"],
                    record.get("target_scope", "run"),
                    record.get("target_id", "__run__"),
                    record.get("value"),
                    record.get("payload_json"),
                    record.get("total_count"),
                    record.get("scored_count"),
                ]
            )

        return self.db_io.executemany(
            """
            INSERT OR REPLACE INTO metric_results (
                job_id,
                config_hash,
                metric_name,
                target_scope,
                target_id,
                value,
                payload_json,
                total_count,
                scored_count,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            batch,
        )

    def get_run_metadata(self, config_hash: str) -> Optional[Dict[str, Any]]:
        """Load run metadata by config_hash (newest job wins), or None if missing."""

        keys = (
            "job_id",
            "config_hash",
            "model_config",
            "model_id",
            "dataset_name",
            "dataset_subset",
            "split",
            "git_commit_hash",
            "task",
            "remarks",
            "status",
            "run_name",
            "project_tags",
            "total_samples",
        )
        row = self.db_io.fetchone(
            f"""
            SELECT {', '.join(keys)}
            FROM {RUN_METADATA_TABLE}
            WHERE config_hash = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [config_hash],
        )
        if row is None:
            return None
        return dict(zip(keys, row, strict=True))

    def update_run_counts(
        self,
        config_hash: str,
        total: int,
        processed: int,
        error: int,
        job_id: Optional[str] = None,
        post_processing_errors: Optional[int] = None,
        cache_hits: Optional[int] = None,
        duration_seconds: Optional[float] = None,
    ) -> bool:
        """Write end-of-run counts onto the job row."""

        if not job_id and config_hash:
            job_id = self._latest_job_for_config(config_hash)
        if not job_id:
            return False
        self.db_io.execute(
            f"""
            UPDATE {RUN_METADATA_TABLE}
            SET total_samples = ?,
                processed_samples = ?,
                success_count = ?,
                error_count = ?,
                post_processing_error_count = COALESCE(?, post_processing_error_count),
                cache_hits = COALESCE(?, cache_hits),
                duration_seconds = COALESCE(?, duration_seconds),
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
            """,
            [
                int(total or 0),
                int(processed or 0),
                int(processed or 0),
                int(error or 0),
                None if post_processing_errors is None else int(post_processing_errors),
                None if cache_hits is None else int(cache_hits),
                None if duration_seconds is None else float(duration_seconds),
                str(job_id),
            ],
        )
        return True

    def _latest_job_for_config(self, config_hash: str) -> Optional[str]:
        """Newest job_id for a config_hash, preferring prod jobs."""

        row = self.db_io.fetchone(
            f"SELECT job_id FROM {RUN_METADATA_TABLE} WHERE config_hash = ? "
            "ORDER BY (run_mode = 'prod') DESC NULLS LAST, "
            "created_at DESC NULLS LAST LIMIT 1",
            [config_hash],
        )
        return str(row[0]) if row else None

    def save_run_metadata(self, run_data: Dict[str, Any]) -> bool:
        """Upsert a job row; lookup by job_id first, else newest job for config_hash."""
        try:
            import json
            import uuid

            project_tags = run_data.get("project_tags")
            if isinstance(project_tags, list):
                project_tags = json.dumps(project_tags)

            job_id = run_data.get("job_id") or None
            config_hash = run_data.get("config_hash") or None

            existing = None
            if job_id:
                row = self.db_io.fetchone(
                    f"SELECT job_id FROM {RUN_METADATA_TABLE} WHERE job_id = ?",
                    [job_id],
                )
                existing = str(row[0]) if row else None
            elif config_hash:
                existing = self._latest_job_for_config(config_hash)
            else:
                logger.error(
                    "save_run_metadata requires job_id or config_hash; skipping"
                )
                return False

            values = {
                "config_hash": config_hash,
                "model_config": run_data.get("model_config"),
                "model_id": run_data.get("model_id"),
                "dataset_name": run_data.get("dataset_name"),
                "dataset_subset": run_data.get("dataset_subset"),
                "split": run_data.get("split") or run_data.get("dataset_split"),
                "git_commit_hash": run_data.get("git_commit_hash"),
                "task": run_data.get("task"),
                "remarks": run_data.get("remarks"),
                "status": run_data.get("status"),
                "run_name": run_data.get("run_name"),
                "project_tags": project_tags,
                "max_samples": run_data.get("max_samples"),
                "run_mode": run_data.get("run_mode"),
                "error_message": run_data.get("error_message"),
            }

            if existing is None:
                new_job_id = job_id or str(uuid.uuid4())
                if values["status"] is None:
                    values["status"] = "running"
                cols = ["job_id", *values.keys()]
                placeholders = ", ".join("?" for _ in cols)
                self.db_io.execute(
                    f"INSERT INTO {RUN_METADATA_TABLE} ({', '.join(cols)}) "
                    f"VALUES ({placeholders})",
                    [new_job_id, *values.values()],
                )
            else:
                set_clause = ", ".join(
                    f"{col} = COALESCE(?, {col})" for col in values
                )
                self.db_io.execute(
                    f"UPDATE {RUN_METADATA_TABLE} "
                    f"SET {set_clause}, updated_at = CURRENT_TIMESTAMP "
                    "WHERE job_id = ?",
                    [*values.values(), existing],
                )
            return True
        except Exception as e:
            logger.info(f"Error saving run metadata: {e}")
            return False


    def get_catalog_entities(self, entity_type: str) -> List[str]:
        try:
            if entity_type == "dataset":
                rows = self.db_io.fetchall(
                    f"SELECT DISTINCT dataset_name FROM {RUN_METADATA_TABLE} "
                    f"WHERE dataset_name IS NOT NULL"
                )
                return [r[0] for r in rows if r[0]]
            if entity_type == "model":
                rows = self.db_io.fetchall(
                    f"SELECT DISTINCT model_id FROM {RUN_METADATA_TABLE} "
                    f"WHERE model_id IS NOT NULL"
                )
                return [r[0] for r in rows if r[0]]
            if entity_type == "model_path":
                rows = self.db_io.fetchall(
                    f"SELECT DISTINCT json_extract_string(model_config, '$.model_path') "
                    f"FROM {RUN_METADATA_TABLE} WHERE model_config IS NOT NULL"
                )
                return [r[0] for r in rows if r[0]]
        except Exception as e:
            logger.info(f"Catalog query failed ({entity_type}): {e}")
        return []

    def update_run_remarks(self, config_hash: str, remarks: Optional[str]) -> bool:
        """Update the remarks field on the newest job for a config."""
        try:
            job_id = self._latest_job_for_config(config_hash)
            if not job_id:
                return False
            self.db_io.execute(
                f"UPDATE {RUN_METADATA_TABLE} SET remarks = ? WHERE job_id = ?",
                [remarks, job_id],
            )
            return True
        except Exception as e:
            logger.info(f"Error updating run remarks: {e}")
            return False

    def remove_run_status(self, config_hash: str) -> bool:
        """Remove the status field from the newest job for a config."""
        try:
            job_id = self._latest_job_for_config(config_hash)
            if not job_id:
                return False
            self.db_io.execute(
                f"UPDATE {RUN_METADATA_TABLE} SET status = NULL WHERE job_id = ?",
                [job_id],
            )
            return True
        except Exception as e:
            logger.info(f"Error removing run status: {e}")
            return False

    def clear_draft_metadata(self, config_hash: str) -> bool:
        """Clear draft-only fields on the newest job for a config."""
        try:
            job_id = self._latest_job_for_config(config_hash)
            if not job_id:
                return False
            self.db_io.execute(
                f"""
                UPDATE {RUN_METADATA_TABLE}
                SET status = NULL,
                    run_name = NULL
                WHERE job_id = ?
            """,
                [job_id],
            )
            return True
        except Exception as e:
            logger.info(f"Error clearing draft metadata: {e}")
            return False

    def close_connections(self):
        """Close all connections and cleanup."""
        logger.info("🔄 Shutting down simple cache...")

        # Close connections
        if self.db_io:
            self.db_io.close_connections()

        logger.info("✅ Simple cache shutdown complete")

    def export_run_samples_to_parquet(
        self,
        *,
        config_hash: str,
        file_path: str | Path,
    ) -> Tuple[str, int]:
        """Export all samples for a run into a Parquet file."""

        run_exists = self.db_io.fetchone(
            f"SELECT 1 FROM {RUN_METADATA_TABLE} WHERE config_hash = ?",
            [config_hash],
        )
        if run_exists is None:
            raise ValueError(f"Run with config_hash={config_hash} not found")

        total_row = self.db_io.fetchone(
            "SELECT COUNT(*) FROM inference_results WHERE config_hash = ?",
            [config_hash],
        )
        total_samples = int(total_row[0]) if total_row else 0
        logger.info("Found %d samples for export", total_samples)

        output_path = Path(file_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_input_select = (
            "ir.model_input" if self._supports_model_input else "NULL AS model_input"
        )

        if self._supports_rubric_table:
            rubric_select = """
                    COALESCE(
                        (
                            SELECT json_group_array(payload)
                            FROM (
                                SELECT json_object(
                                    'question_index', rer.question_index,
                                    'question_score', rer.question_score,
                                    'rubric_evaluations', rer.rubric_evaluations_json,
                                    'metric_metadata', rer.metric_metadata_json
                                ) AS payload
                                FROM rubric_evaluation_results rer
                                WHERE rer.config_hash = ir.config_hash
                                  AND rer.cache_key = ir.cache_key
                                ORDER BY rer.question_index
                            ) ordered_rer
                        ),
                        '[]'
                    )
            """.strip()
        else:
            rubric_select = "'[]'"

        destination = str(output_path).replace("'", "''")

        export_sql = f"""
            COPY (
                SELECT
                    ir.config_hash,
                    ir.cache_key,
                    ir.dataset_name,
                    ir.dataset_row_hash,
                    ir.dataset_row_metadata,
                    {model_input_select},
                    ir.model_output,
                    ir.tool_trace,
                    ir.ground_truth_output,
                    ir.success,
                    CAST(ir.created_at AS TIMESTAMP) AS created_at,
                    CAST(ir.updated_at AS TIMESTAMP) AS updated_at,
                    rm.model_id,
                    rm.dataset_name AS run_dataset_name,
                    rm.dataset_subset AS run_dataset_subset,
                    rm.split AS run_split,
                    rm.git_commit_hash,
                    rm.task,
                    CAST(rm.created_at AS TIMESTAMP) AS run_created_at,
                    (
                        SELECT MAX(CAST(updated_at AS TIMESTAMP))
                        FROM inference_results ir2
                        WHERE ir2.config_hash = ir.config_hash
                    ) AS run_last_updated,
                    {rubric_select} AS rubric_evaluations
                FROM inference_results ir
                LEFT JOIN {RUN_METADATA_TABLE} rm
                  ON ir.config_hash = rm.config_hash
                WHERE ir.config_hash = ?
                ORDER BY created_at ASC
            ) TO '{destination}' (FORMAT PARQUET);
        """

        self.db_io.execute(export_sql, [config_hash])
        return str(output_path), total_samples

    def delete_run(self, config_hash: str) -> None:
        """Delete ONE job by job_id or config_hash (newest job).

        Removes the job row, its metric rows, and its run_samples links.
        ``inference_results`` is the shared generation cache, never deleted
        here (caching spec C7/R4).
        """
        row = self.db_io.fetchone(
            f"SELECT job_id FROM {RUN_METADATA_TABLE} WHERE job_id = ?",
            [config_hash],
        )
        job_id = str(row[0]) if row else self._latest_job_for_config(config_hash)
        if not job_id:
            return

        if self._check_table_exists("metric_results"):
            self.db_io.execute(
                "DELETE FROM metric_results WHERE job_id = ?", [job_id]
            )
        if self._check_table_exists("run_samples"):
            self.db_io.execute(
                "DELETE FROM run_samples WHERE job_id = ?", [job_id]
            )
        self.db_io.execute(
            f"DELETE FROM {RUN_METADATA_TABLE} WHERE job_id = ?", [job_id]
        )
