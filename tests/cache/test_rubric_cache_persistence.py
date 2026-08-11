import json

import pytest

from karma.cache import CacheManager
from karma.cache.constants import RUN_METADATA_TABLE
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.data_models.model_meta import ModelMeta, ModelType


@pytest.fixture()
def model_meta():
    return ModelMeta(
        name="test-model",
        loader_class="karma.models.base_model_abs.BaseModel",
        loader_kwargs={},
        default_eval_kwargs={},
        model_type=ModelType.TEXT_GENERATION,
    )


def test_rubric_results_persisted(tmp_path, monkeypatch, model_meta):
    db_path = tmp_path / "rubric-cache.duckdb"
    monkeypatch.setenv("KARMA_CACHE_PATH", str(db_path))
    monkeypatch.setenv("KARMA_CACHE_TYPE", "duckdb")

    cache_manager = CacheManager(dataset_name="test/dataset", model_config=model_meta)

    sample = DataLoaderIterable(input="prompt text")
    serialized_sample = sample.model_dump()
    identifiers = cache_manager.get_cache_identifiers_for_sample(serialized_sample)

    record = {
        "cache_key": identifiers["cache_key"],
        "dataset_row_hash": identifiers["dataset_row_hash"],
        "dataset_name": "test/dataset",
        "config_hash": cache_manager.config_hash,
        "metric_name": "rubric_evaluation",
        "question_index": 0,
        "question_score": 0.75,
        "rubric_evaluations": [
            {
                "criteria_met": True,
                "explanation": "meets guidance",
                "rubric": {
                    "criterion": "criterion text",
                    "points": 1,
                    "tags": ["safety"],
                },
            }
        ],
        "metric_metadata": {
            "provider": "openai",
            "model_identifier": "gpt-4o-mini",
            "batch_size": 3,
        },
    }

    assert cache_manager.batch_save_rubric_question_results([record])

    row = cache_manager.cache_io.db_io.fetchone(
        """
        SELECT dataset_name, metric_name, question_index, question_score, rubric_evaluations_json, metric_metadata_json
        FROM rubric_evaluation_results
        WHERE cache_key = ?
        """,
        [record["cache_key"]],
    )

    assert row is not None
    assert row[0] == record["dataset_name"]
    assert row[1] == record["metric_name"]
    assert row[2] == record["question_index"]
    assert pytest.approx(row[3]) == record["question_score"]

    stored_payload = json.loads(row[4])
    assert stored_payload[0]["criteria_met"] is True
    assert stored_payload[0]["rubric"]["criterion"] == "criterion text"

    stored_metadata = json.loads(row[5])
    assert stored_metadata["provider"] == "openai"
    assert stored_metadata["model_identifier"] == "gpt-4o-mini"

    cache_manager.close_connections()


def test_inference_results_store_model_input(tmp_path, monkeypatch, model_meta):
    db_path = tmp_path / "rubric-cache.duckdb"
    monkeypatch.setenv("KARMA_CACHE_PATH", str(db_path))
    monkeypatch.setenv("KARMA_CACHE_TYPE", "duckdb")

    cache_manager = CacheManager(dataset_name="test/dataset", model_config=model_meta)

    sample = DataLoaderIterable(input="Patient presents with sore throat")
    serialized_sample = sample.model_dump()
    identifiers = cache_manager.get_cache_identifiers_for_sample(serialized_sample)

    batch_row = {
        "sample": serialized_sample,
        "prediction": "Recommend follow-up",
        "expected_output": "Recommend follow-up",
        "success": True,
    }

    assert cache_manager.batch_save_rows([batch_row], dataset_name="test/dataset")

    row = cache_manager.cache_io.db_io.fetchone(
        """
        SELECT model_input
        FROM inference_results
        WHERE cache_key = ?
        """,
        [identifiers["cache_key"]],
    )

    assert row is not None
    stored_input = json.loads(row[0])
    assert stored_input["input"] == "Patient presents with sore throat"

    cache_manager.close_connections()


def test_refresh_cache_creates_new_config_hash(tmp_path, monkeypatch, model_meta):
    db_path = tmp_path / "rubric-cache.duckdb"
    monkeypatch.setenv("KARMA_CACHE_PATH", str(db_path))
    monkeypatch.setenv("KARMA_CACHE_TYPE", "duckdb")

    base_manager = CacheManager(dataset_name="test/dataset", model_config=model_meta)
    refreshed_manager = CacheManager(
        dataset_name="test/dataset",
        model_config=model_meta,
        run_instance_id="run-123",
    )

    assert base_manager.config_hash != refreshed_manager.config_hash

    base_manager.close_connections()
    refreshed_manager.close_connections()


def test_run_metadata_persists_dataset_details(tmp_path, monkeypatch, model_meta):
    db_path = tmp_path / "run-metadata-cache.duckdb"
    monkeypatch.setenv("KARMA_CACHE_PATH", str(db_path))
    monkeypatch.setenv("KARMA_CACHE_TYPE", "duckdb")

    cache_manager = CacheManager(dataset_name="test/dataset", model_config=model_meta)
    cache_manager.record_dataset_metadata(
        dataset_subset="subset-a", split="validation"
    )

    row = cache_manager.cache_io.db_io.fetchone(
        f"""
        SELECT dataset_subset, split
        FROM {RUN_METADATA_TABLE}
        WHERE config_hash = ?
        """,
        [cache_manager.config_hash],
    )

    assert row == ("subset-a", "validation")

    cache_manager.close_connections()
