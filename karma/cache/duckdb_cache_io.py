from typing import Any, Dict, Optional, List

from karma.io.duckdb_io import DuckDBIO


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

    def __init__(self, db_path: str = "cache/benchmark_cache.duckdb"):
        """Initialize simple cache IO."""
        # Initialize DuckDB (persistent storage) with hardcoded path
        self.db_io = DuckDBIO(db_path=db_path)
        print("✅ DuckDB persistent storage connected")
        # Initialize schema for benchmark caching
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema for benchmark caching."""
        # Create inference results table
        self.db_io.execute("""
            CREATE TABLE IF NOT EXISTS inference_results (
                cache_key VARCHAR(64) PRIMARY KEY,
                dataset_row_hash VARCHAR(64) NOT NULL,
                dataset_name VARCHAR(64) NOT NULL,
                model_output TEXT NOT NULL,
                model_output_reasoning TEXT,
                ground_truth_output TEXT,
                ground_truth_reasoning TEXT,
                config_hash VARCHAR(64) NOT NULL,
                success BOOLEAN NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create run metadata table
        self.db_io.execute("""
            CREATE TABLE IF NOT EXISTS run_metadata (
                config_hash VARCHAR(64) PRIMARY KEY,
                model_config TEXT NOT NULL,
                model_id VARCHAR(255) NOT NULL,
                dataset_name VARCHAR(255) NOT NULL,
                task TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indices for performance - especially for config_hash lookups
        self.db_io.execute("""
            CREATE INDEX IF NOT EXISTS idx_inference_config_hash 
            ON inference_results(config_hash)
        """)
        self.db_io.execute("""
            CREATE INDEX IF NOT EXISTS idx_inference_cache_key 
            ON inference_results(cache_key)
        """)

    def get_inference_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get a cached inference result by cache key from DuckDB."""
        result = self.db_io.fetchone(
            """
            SELECT cache_key, dataset_row_hash, model_output, 
                   model_output_reasoning, ground_truth_output, 
                   ground_truth_reasoning, config_hash, success
            FROM inference_results 
            WHERE cache_key = ?
        """,
            [cache_key],
        )

        if result:
            return {
                "cache_key": result[0],
                "dataset_row_hash": result[1],
                "model_output": result[2],
                "model_output_reasoning": result[3],
                "ground_truth_output": result[4],
                "ground_truth_reasoning": result[5],
                "config_hash": result[6],
                "success": result[7],
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
            SELECT cache_key, dataset_row_hash, model_output, 
                   model_output_reasoning, ground_truth_output, 
                   ground_truth_reasoning, config_hash, success
            FROM inference_results 
            WHERE cache_key IN ({placeholders})
        """,
            cache_keys,
        )

        results = {}
        for row in results_data:
            results[row[0]] = {
                "cache_key": row[0],
                "dataset_row_hash": row[1],
                "model_output": row[2],
                "model_output_reasoning": row[3],
                "ground_truth_output": row[4],
                "ground_truth_reasoning": row[5],
                "config_hash": row[6],
                "success": row[7],
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
                    data["dataset_row_hash"],
                    data["model_output"],
                    data.get("model_output_reasoning"),
                    data.get("ground_truth_output"),
                    data.get("ground_truth_reasoning"),
                    data["config_hash"],
                    data["success"],
                ]
            )

        return self.db_io.executemany(
            """
            INSERT OR REPLACE INTO inference_results 
            (cache_key, dataset_row_hash, model_output, model_output_reasoning,
             ground_truth_output, ground_truth_reasoning, config_hash, success, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            batch_data,
        )

    def save_run_metadata(self, run_data: Dict[str, Any]) -> bool:
        """Save run metadata to DuckDB."""
        try:
            self.db_io.execute(
                """
                INSERT OR IGNORE INTO run_metadata 
                (config_hash, model_config, model_id, dataset_name, task)
                VALUES (?, ?, ?, ?, ?)
            """,
                [
                    run_data["config_hash"],
                    run_data["model_config"],
                    run_data["model_id"],
                    run_data["dataset_name"],
                    run_data.get("task"),
                ],
            )
            return True
        except Exception as e:
            print(f"Error saving run metadata: {e}")
            return False

    def close_connections(self):
        """Close all connections and cleanup."""
        print("🔄 Shutting down simple cache...")

        # Close connections
        if self.db_io:
            self.db_io.close_connections()

        print("✅ Simple cache shutdown complete")
