import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
from boto3.dynamodb.conditions import Attr, Key


# Expected DynamoDB GSIs. Create them with these names or override via env vars:
# - KARMA_DYNAMO_CONFIG_HASH_INDEX  (defaults to "config_hash-index")
# - KARMA_DYNAMO_ROW_HASH_INDEX     (optional, e.g. "dataset_row_hash-index")
# - KARMA_DYNAMO_RUN_INDEX          (optional, e.g. "dataset_name-created_at-index")
DEFAULT_CONFIG_HASH_INDEX = "config_hash-index"


def _convert_floats_to_decimal(obj: Any) -> Any:
    """Recursively convert float values to Decimal for DynamoDB compatibility."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: _convert_floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_floats_to_decimal(item) for item in obj]
    return obj


def _convert_decimals_to_float(obj: Any) -> Any:
    """Recursively convert Decimal values to float for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_decimals_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_decimals_to_float(item) for item in obj]
    return obj


class DynamoDBCacheIO:
    """
    Simplified cache IO using DynamoDB with direct operations.

    Architecture:
    - DynamoDB: Serverless NoSQL database for all cache data
    - Direct operations: Fast key-value lookups and batch operations
    - Pay-per-request: No capacity planning needed

    Features:
    - Fast key-value reads and writes
    - Batch operations for efficiency
    - Auto-scaling and managed infrastructure
    """

    def __init__(self):
        """Initialize DynamoDB cache IO."""
        # Initialize DynamoDB
        self.table_name = "parrotlet_omni_benchmark_cache"
        self.region_name = os.getenv("AWS_REGION", "ap-south-1")
        self.config_hash_index = (
            os.getenv("KARMA_DYNAMO_CONFIG_HASH_INDEX") or DEFAULT_CONFIG_HASH_INDEX
        )
        self.row_hash_index = "dataset_row_hash-config_hash-index"
        self.run_index_name = "config_hash-created_at-index"

        if boto3 is None:  # pragma: no cover - triggered when dependency missing
            raise RuntimeError(
                "boto3 is required for DynamoDB cache backend. Install with "
                "`pip install boto3`."
            )
        aws_access_key_id = os.getenv("AWS_DYNAMO_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_DYNAMO_SECRET_ACCESS_KEY")
        if aws_access_key_id and aws_secret_access_key:
            self.dynamodb = boto3.resource("dynamodb", region_name=self.region_name, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
            self.client = boto3.client("dynamodb", region_name=self.region_name, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
        else:
            self.dynamodb = boto3.resource("dynamodb", region_name=self.region_name)
            self.client = boto3.client("dynamodb", region_name=self.region_name)

        # Initialize tables
        self._init_tables()
        print(
            f"✅ DynamoDB connected - Table: {self.table_name}, Region: {self.region_name}"
        )

    def _init_tables(self):
        """Initialize DynamoDB tables for benchmark caching."""

        self.table = self._load_table(self.table_name, required=True)

        # Check if metadata table exists
        metadata_table_name = f"{self.table_name}_run_metadata"
        self.metadata_table = self._load_table(metadata_table_name, required=True)

        rubric_table_name = f"{self.table_name}_rubric_evaluation_results_v2"
        self.rubric_table = self._load_table(rubric_table_name, required=False)

        metric_table_name = f"{self.table_name}_metric_results"
        self.metric_table = self._load_table(metric_table_name, required=False)


        catalog_table_name = f"{self.table_name}_catalog"
        self.catalog_table = self._load_table(catalog_table_name, required=False)

    def _load_table(self, table_name: str, *, required: bool) -> Optional[Any]:
        table = self.dynamodb.Table(table_name)
        try:
            table.load()
        except Exception:
            if required:
                raise
            return None
        return table

    def get_inference_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get a cached inference result by cache key from DynamoDB."""
        try:
            response = self.table.get_item(Key={"cache_key": cache_key})

            if "Item" in response:
                item = response["Item"]
                return item
            return None
        except Exception as e:
            print(f"Error getting inference result: {e}")
            return None

    def batch_get_inference_results(
        self, cache_keys: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Get multiple cached inference results by cache keys from DynamoDB."""
        if not cache_keys:
            return {}

        results = {}

        try:
            # DynamoDB batch_get_item has a limit of 100 items
            for i in range(0, len(cache_keys), 100):
                batch_keys = cache_keys[i : i + 100]
                keys = [{"cache_key": key} for key in batch_keys]

                response = self.dynamodb.batch_get_item(
                    RequestItems={self.table_name: {"Keys": keys}}
                )

                if self.table_name in response["Responses"]:
                    for item in response["Responses"][self.table_name]:
                        results[item["cache_key"]] = item

            # Add None entries for missing keys
            for cache_key in cache_keys:
                if cache_key not in results:
                    results[cache_key] = None

            return results

        except Exception as e:
            print(f"Error batch getting inference results: {e}")
            # Return empty results for all keys on error
            return {key: None for key in cache_keys}  # type: ignore

    def batch_save_inference_results(
        self, inference_data_list: List[Dict[str, Any]]
    ) -> bool:
        """Save multiple inference results to DynamoDB."""

        try:
            # Use DynamoDB batch writer for efficient batch operations
            with self.table.batch_writer() as batch:
                for data in inference_data_list:
                    data["created_at"] = int(time.time())
                    batch.put_item(Item=_convert_floats_to_decimal(data))

            return True

        except Exception as e:
            print(f"Error batch saving inference results: {e}")
            return False

    def batch_save_metric_results(self, records: List[Dict[str, Any]]) -> bool:
        """Persist generic metric outputs when DynamoDB backing table is available."""

        if not records:
            return True

        if (
            getattr(self, "metric_table", None) is None
        ):  # pragma: no cover - optional table missing
            return True

        timestamp = int(time.time())

        try:
            with self.metric_table.batch_writer() as batch:  # type: ignore[attr-defined]
                for record in records:
                    item = {**record}
                    item["created_at"] = timestamp
                    item["updated_at"] = timestamp

                    metric_name = str(item.get("metric_name", ""))
                    target_scope = str(item.get("target_scope", "run"))
                    target_id = str(item.get("target_id", "__run__"))

                    if not metric_name:
                        raise ValueError(
                            "metric_name is required for metric persistence"
                        )

                    item["metric_name"] = metric_name
                    item["target_scope"] = target_scope
                    item["target_id"] = target_id
                    item["target_sort_key"] = item.get(
                        "target_sort_key",
                        f"{target_scope}#{target_id}#{metric_name}",
                    )

                    value = item.get("value")
                    if value is not None:
                        item["value"] = Decimal(str(value))

                    for key in ("total_count", "scored_count"):
                        if item.get(key) is not None:
                            item[key] = int(item[key])

                    batch.put_item(Item=item)

            return True

        except Exception as exc:  # pragma: no cover - network/service errors
            print(f"Error saving metric results: {exc}")
            return False

    def batch_save_rubric_question_results(self, records: List[Dict[str, Any]]) -> bool:
        """Persist rubric evaluation records when a DynamoDB backing table exists."""

        if not records:
            return True

        if (
            getattr(self, "rubric_table", None) is None
        ):  # pragma: no cover - optional table missing
            return True

        timestamp = int(time.time())

        try:
            with self.rubric_table.batch_writer() as batch:  # type: ignore[attr-defined]
                for record in records:
                    metric_name = record["metric_name"]
                    question_index = int(record["question_index"])
                    item: Dict[str, Any] = {
                        "cache_key": record["cache_key"],
                        "metric_question_key": f"{metric_name}#{question_index}",
                        "dataset_name": record["dataset_name"],
                        "config_hash": record["config_hash"],
                        "dataset_row_hash": record["dataset_row_hash"],
                        "metric_name": metric_name,
                        "question_index": question_index,
                        "rubric_evaluations_json": json.dumps(
                            record.get("rubric_evaluations", {})
                        ),
                        "metric_metadata_json": json.dumps(
                            record.get("metric_metadata", {})
                        ),
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    }

                    question_score = record.get("question_score")
                    if question_score is not None:
                        item["question_score"] = Decimal(str(question_score))

                    batch.put_item(Item=item)

            return True

        except Exception as exc:  # pragma: no cover - network/service errors
            print(f"Error saving rubric results: {exc}")
            return False

    def get_run_metadata(self, config_hash: str) -> Optional[Dict[str, Any]]:
        """Load a single run metadata item by config_hash, or None if missing."""

        try:
            response = self.metadata_table.get_item(Key={"config_hash": config_hash})
        except Exception as exc:  # pragma: no cover - network/service errors
            print(f"Error loading run metadata: {exc}")
            return None
        item = response.get("Item") if response else None
        if not item:
            return None
        return _convert_decimals_to_float(item)

    @staticmethod
    def _extract_model_path(model_config: Any) -> Optional[str]:
        if not model_config:
            return None
        if isinstance(model_config, str):
            try:
                import json as _json
                model_config = _json.loads(model_config)
            except Exception:
                return None
        if isinstance(model_config, dict):
            return model_config.get("model_path") or None
        return None

    def save_run_metadata(self, run_data: Dict[str, Any]) -> bool:
        """Save run metadata to DynamoDB, including created_at and updated_at."""
        try:
            normalized_time = int(time.time())

            attr_names = {
                "#updated_at": "updated_at",
                "#created_at": "created_at",
            }
            attr_values: Dict[str, Any] = {
                ":updated_at": normalized_time,
            }

            # Attempt to use an existing created_at, else set now
            created_at_value = normalized_time
            attr_values[":created_at"] = created_at_value

            update_parts = [
                "#updated_at = :updated_at",
                "#created_at = if_not_exists(#created_at, :created_at)",
            ]

            for field in (
                "model_config",
                "model_id",
                "dataset_name",
                "git_commit_hash",
                "task",
                "dataset_subset",
                "split",
                "remarks",
                "status",
                "run_name",
                "wandb_run_id",
                "instance_type",
                "project_tags",
            ):
                value = run_data.get(field)
                if value is None:
                    continue
                name_key = f"#{field}"
                value_key = f":{field}"
                attr_names[name_key] = field
                attr_values[value_key] = value
                update_parts.append(f"{name_key} = {value_key}")

            self.metadata_table.update_item(
                Key={"config_hash": run_data["config_hash"]},
                UpdateExpression="SET " + ", ".join(update_parts),
                ExpressionAttributeNames=attr_names,
                ExpressionAttributeValues=attr_values,
            )


            if dataset_name := run_data.get("dataset_name"):
                self.upsert_catalog_entity("dataset", dataset_name)
            if model_id := run_data.get("model_id"):
                self.upsert_catalog_entity("model", model_id)
            if model_path := self._extract_model_path(run_data.get("model_config")):
                self.upsert_catalog_entity("model_path", model_path)

            return True
        except Exception as e:
            print(f"Error saving run metadata: {e}")
            return False

    def upsert_catalog_entity(self, entity_type: str, entity_value: str) -> None:
        if not entity_value or not getattr(self, "catalog_table", None):
            return
        try:
            self.catalog_table.put_item(
                Item={"entity_type": entity_type, "entity_value": entity_value}
            )
        except Exception as e:
            print(f"Catalog upsert skipped ({entity_type}={entity_value!r}): {e}")

    def get_catalog_entities(self, entity_type: str) -> List[str]:
        if not getattr(self, "catalog_table", None):
            return []
        try:
            results: List[str] = []
            last_key = None
            while True:
                kwargs: Dict[str, Any] = {
                    "KeyConditionExpression": Key("entity_type").eq(entity_type)
                }
                if last_key:
                    kwargs["ExclusiveStartKey"] = last_key
                resp = self.catalog_table.query(**kwargs)
                results.extend(
                    item["entity_value"] for item in resp.get("Items", [])
                )
                last_key = resp.get("LastEvaluatedKey")
                if not last_key:
                    break
            return results
        except Exception as e:
            print(f"Catalog query failed ({entity_type}): {e}")
            return []

    def update_run_remarks(self, config_hash: str, remarks: Optional[str]) -> bool:
        """Update the remarks field for a run."""
        try:
            timestamp = int(time.time())
            self.metadata_table.update_item(
                Key={"config_hash": config_hash},
                UpdateExpression="SET #remarks = :remarks, #updated_at = :updated_at",
                ExpressionAttributeNames={
                    "#remarks": "remarks",
                    "#updated_at": "updated_at",
                },
                ExpressionAttributeValues={
                    ":remarks": remarks,
                    ":updated_at": timestamp,
                },
            )
            return True
        except Exception as e:
            print(f"Error updating run remarks: {e}")
            return False

    def update_run_tags(self, config_hash: str, project_tags: list) -> bool:
        """Update the project_tags field for a run."""
        try:
            timestamp = int(time.time())
            self.metadata_table.update_item(
                Key={"config_hash": config_hash},
                UpdateExpression="SET #project_tags = :project_tags, #updated_at = :updated_at",
                ExpressionAttributeNames={
                    "#project_tags": "project_tags",
                    "#updated_at": "updated_at",
                },
                ExpressionAttributeValues={
                    ":project_tags": project_tags,
                    ":updated_at": timestamp,
                },
            )
            return True
        except Exception as e:
            print(f"Error updating run tags: {e}")
            return False

    def remove_run_status(self, config_hash: str) -> bool:
        """Remove the status field from a run, making it a normal run."""
        try:
            timestamp = int(time.time())
            self.metadata_table.update_item(
                Key={"config_hash": config_hash},
                UpdateExpression="REMOVE #status SET #updated_at = :updated_at",
                ExpressionAttributeNames={
                    "#status": "status",
                    "#updated_at": "updated_at",
                },
                ExpressionAttributeValues={
                    ":updated_at": timestamp,
                },
            )
            return True
        except Exception as e:
            print(f"Error removing run status: {e}")
            return False

    def clear_draft_metadata(self, config_hash: str) -> bool:
        """Clear draft-only fields while preserving historical run data."""
        try:
            timestamp = int(time.time())
            self.metadata_table.update_item(
                Key={"config_hash": config_hash},
                UpdateExpression=(
                    "REMOVE #status, #run_name, #wandb_run_id, #instance_type "
                    "SET #updated_at = :updated_at"
                ),
                ExpressionAttributeNames={
                    "#status": "status",
                    "#run_name": "run_name",
                    "#wandb_run_id": "wandb_run_id",
                    "#instance_type": "instance_type",
                    "#updated_at": "updated_at",
                },
                ExpressionAttributeValues={
                    ":updated_at": timestamp,
                },
            )
            return True
        except Exception as e:
            print(f"Error clearing draft metadata: {e}")
            return False

    def close_connections(self):
        """Close DynamoDB connections."""
        print("✅ DynamoDB connections closed")

    # DynamoDB-specific helpers

    def _query_items_by_config_hash(
        self,
        table,
        config_hash: str,
        *,
        projection_expression: Optional[str] = None,
        index_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query or scan a table for all items matching the config hash."""

        if table is None:
            return []

        items: List[Dict[str, Any]] = []
        last_key: Optional[Dict[str, Any]] = None
        resolved_index = index_name or self.config_hash_index
        use_query = bool(resolved_index) and Key is not None

        while True:
            if use_query:
                kwargs: Dict[str, Any] = {
                    "IndexName": resolved_index,
                    "KeyConditionExpression": Key("config_hash").eq(config_hash),
                }
            else:
                kwargs = {}
                if Attr is not None:
                    kwargs["FilterExpression"] = Attr("config_hash").eq(config_hash)

            if projection_expression:
                kwargs["ProjectionExpression"] = projection_expression

            if last_key:
                kwargs["ExclusiveStartKey"] = last_key

            try:
                response = table.query(**kwargs) if use_query else table.scan(**kwargs)
            except Exception:
                if use_query:
                    use_query = False
                    last_key = None
                    continue
                raise

            items.extend(response.get("Items", []))
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break

        return items

    @staticmethod
    def _normalize_timestamp(value: Any) -> Optional[str]:
        """Convert DynamoDB timestamp representations to ISO 8601 strings."""

        if value is None:
            return None

        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            else:
                value = value.astimezone(timezone.utc)
            return value.isoformat()

        if isinstance(value, Decimal):
            value = float(value)

        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                return None
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None

            numeric_candidate = stripped.lstrip("-").replace(".", "", 1)
            if numeric_candidate.isdigit():
                try:
                    numeric_value = float(stripped)
                except ValueError:
                    numeric_value = None
                else:
                    if math.isfinite(numeric_value):
                        return datetime.fromtimestamp(
                            numeric_value, tz=timezone.utc
                        ).isoformat()

            return stripped

        return str(value)

    def _load_rubric_evaluations(self, config_hash: str) -> Dict[str, str]:
        """Aggregate rubric evaluation payloads per cache key."""

        records = self._query_items_by_config_hash(self.rubric_table, config_hash)
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            cache_key = record.get("cache_key")
            if not cache_key:
                continue
            grouped[cache_key].append(
                {
                    "question_index": record.get("question_index"),
                    "question_score": record.get("question_score"),
                    "rubric_evaluations": record.get("rubric_evaluations_json"),
                    "metric_metadata": record.get("metric_metadata_json"),
                }
            )

        aggregated: Dict[str, str] = {}
        for cache_key, payloads in grouped.items():
            payloads.sort(key=lambda item: item.get("question_index", 0))
            aggregated[cache_key] = json.dumps(_convert_decimals_to_float(payloads))
        return aggregated

    def export_run_samples_to_parquet(
        self,
        *,
        config_hash: str,
        file_path: str | Path,
    ) -> Tuple[str, int]:
        """Export cached run samples to Parquet using an in-memory DuckDB writer."""

        run_response = self.metadata_table.get_item(Key={"config_hash": config_hash})
        run_item = run_response.get("Item") if run_response else None
        if not run_item:
            raise ValueError(f"Run with config_hash={config_hash} not found")

        inference_items = self._query_items_by_config_hash(self.table, config_hash)
        rubric_lookup = self._load_rubric_evaluations(config_hash)

        output_path = Path(file_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        run_created_at = self._normalize_timestamp(run_item.get("created_at"))
        run_last_updated = self._normalize_timestamp(run_item.get("last_updated"))

        prepared_rows: List[Dict[str, Any]] = []
        last_updated_candidates: List[Optional[str]] = []

        for item in inference_items:
            cache_key = item.get("cache_key")
            if not cache_key:
                continue

            metadata = item.get("dataset_row_metadata")
            if isinstance(metadata, (dict, list)):
                metadata_text = json.dumps(_convert_decimals_to_float(metadata))
            else:
                metadata_text = metadata

            model_input = item.get("model_input")
            if isinstance(model_input, (dict, list)):
                model_input = json.dumps(_convert_decimals_to_float(model_input))

            created_at = self._normalize_timestamp(item.get("created_at"))
            updated_at = self._normalize_timestamp(item.get("updated_at"))
            last_updated_candidates.append(updated_at)

            success = item.get("success")
            if isinstance(success, str):
                success = success.lower() in {"true", "1", "yes"}

            prepared_rows.append(
                {
                    "config_hash": item.get("config_hash", config_hash),
                    "cache_key": cache_key,
                    "dataset_name": item.get("dataset_name"),
                    "dataset_row_hash": item.get("dataset_row_hash"),
                    "dataset_row_metadata": metadata_text,
                    "model_input": model_input,
                    "model_output": item.get("model_output"),
                    "ground_truth_output": item.get("ground_truth_output"),
                    "success": bool(success),
                    "created_at": created_at,
                    "updated_at": updated_at,
                }
            )

        inferred_last_updated = max(
            (ts for ts in last_updated_candidates if ts is not None),
            default=None,
        )
        resolved_run_last_updated = run_last_updated or inferred_last_updated

        prepared_rows.sort(
            key=lambda row: (
                row["created_at"] or "",
                row["cache_key"],
            )
        )

        run_dataset_subset = run_item.get("dataset_subset")
        run_split = run_item.get("split")

        if (run_dataset_subset is None or run_split is None) and run_item.get(
            "model_config"
        ):
            try:
                model_config = (
                    json.loads(run_item["model_config"])
                    if isinstance(run_item["model_config"], str)
                    else run_item["model_config"]
                )
            except Exception:
                model_config = {}
            else:
                if run_dataset_subset is None:
                    run_dataset_subset = model_config.get("dataset_subset")
                if run_split is None:
                    run_split = model_config.get("split")

        rows: List[Tuple[Any, ...]] = []
        for row in prepared_rows:
            rows.append(
                (
                    row["config_hash"],
                    row["cache_key"],
                    row["dataset_name"],
                    row["dataset_row_hash"],
                    row["dataset_row_metadata"],
                    row["model_input"],
                    row["model_output"],
                    row["ground_truth_output"],
                    row["success"],
                    row["created_at"],
                    row["updated_at"],
                    run_item.get("model_id"),
                    run_item.get("dataset_name"),
                    run_dataset_subset,
                    run_split,
                    run_item.get("task"),
                    run_created_at,
                    resolved_run_last_updated,
                    rubric_lookup.get(row["cache_key"], "[]"),
                )
            )

        total_samples = len(rows)

        import duckdb  # Imported lazily to avoid dependency when unused

        conn = duckdb.connect(database=":memory:")
        try:
            conn.execute(
                """
                CREATE TABLE export (
                    config_hash VARCHAR,
                    cache_key VARCHAR,
                    dataset_name VARCHAR,
                    dataset_row_hash VARCHAR,
                    dataset_row_metadata VARCHAR,
                    model_input VARCHAR,
                    model_output VARCHAR,
                    ground_truth_output VARCHAR,
                    success BOOLEAN,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    run_model_id VARCHAR,
                    run_dataset_name VARCHAR,
                    run_dataset_subset VARCHAR,
                    run_split VARCHAR,
                    run_task VARCHAR,
                    run_created_at TIMESTAMP,
                    run_last_updated TIMESTAMP,
                    rubric_evaluations VARCHAR
                )
                """
            )
            if rows:
                conn.executemany(
                    """
                    INSERT INTO export VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    rows,
                )

            destination = str(output_path).replace("'", "''")
            conn.execute(f"COPY export TO '{destination}' (FORMAT PARQUET)")
        finally:
            conn.close()

        return str(output_path), total_samples

    def delete_run(self, config_hash: str) -> None:
        """
        Delete all data associated with a run from DynamoDB.

        This includes:
        - All inference results (data points/samples) from main cache table
        - All rubric evaluation results from rubric table
        - All metric results from metric table
        - Run metadata from metadata table

        Args:
            config_hash: The configuration hash identifying the run to delete
        """
        # Delete all inference results (main cache table)
        # Use query on GSI to get all cache keys for this config_hash
        response = self.table.query(
            IndexName=self.config_hash_index,
            KeyConditionExpression=Key("config_hash").eq(config_hash),
            ProjectionExpression="cache_key",
        )

        items_to_delete = response.get("Items", [])

        # Handle pagination if there are more results
        while "LastEvaluatedKey" in response:
            response = self.table.query(
                IndexName=self.config_hash_index,
                KeyConditionExpression=Key("config_hash").eq(config_hash),
                ProjectionExpression="cache_key",
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            items_to_delete.extend(response.get("Items", []))

        # Batch delete items from main cache table
        if items_to_delete:
            # DynamoDB batch_write can handle max 25 items at a time
            for i in range(0, len(items_to_delete), 25):
                batch = items_to_delete[i:i + 25]
                with self.table.batch_writer() as batch_writer:
                    for item in batch:
                        batch_writer.delete_item(Key={"cache_key": item["cache_key"]})

        # Delete rubric evaluation results if table exists
        if self.rubric_table:
            response = self.rubric_table.query(
                IndexName="config_hash-index",
                KeyConditionExpression=Key("config_hash").eq(config_hash),
                ProjectionExpression="cache_key, metric_question_key",
            )

            rubric_items_to_delete = response.get("Items", [])

            while "LastEvaluatedKey" in response:
                response = self.rubric_table.query(
                    IndexName="config_hash-index",
                    KeyConditionExpression=Key("config_hash").eq(config_hash),
                    ProjectionExpression="cache_key, metric_question_key",
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                )
                rubric_items_to_delete.extend(response.get("Items", []))

            # Batch delete rubric items
            if rubric_items_to_delete:
                for i in range(0, len(rubric_items_to_delete), 25):
                    batch = rubric_items_to_delete[i:i + 25]
                    with self.rubric_table.batch_writer() as batch_writer:
                        for item in batch:
                            batch_writer.delete_item(
                                Key={
                                    "cache_key": item["cache_key"],
                                    "metric_question_key": item["metric_question_key"],
                                }
                            )

        # Delete metric results if table exists
        if self.metric_table:
            response = self.metric_table.query(
                IndexName="config_hash-index",
                KeyConditionExpression=Key("config_hash").eq(config_hash),
                ProjectionExpression="config_hash, target_sort_key",
            )

            metric_items_to_delete = response.get("Items", [])

            while "LastEvaluatedKey" in response:
                response = self.metric_table.query(
                    IndexName="config_hash-index",
                    KeyConditionExpression=Key("config_hash").eq(config_hash),
                    ProjectionExpression="config_hash, target_sort_key",
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                )
                metric_items_to_delete.extend(response.get("Items", []))

            # Batch delete metric items
            if metric_items_to_delete:
                for i in range(0, len(metric_items_to_delete), 25):
                    batch = metric_items_to_delete[i:i + 25]
                    with self.metric_table.batch_writer() as batch_writer:
                        for item in batch:
                            batch_writer.delete_item(
                                Key={
                                    "config_hash": item["config_hash"],
                                    "target_sort_key": item["target_sort_key"],
                                }
                            )

        # Delete run metadata
        self.metadata_table.delete_item(Key={"config_hash": config_hash})
