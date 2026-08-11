"""Read-path reuse rules (stored blank raw reusable, NULL/model_error not)
and ground-truth refresh when the dataset's expected output changed."""

import pytest

from karma.cache import CacheManager
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


class _FakeCacheIO:
    """In-memory cache_io capturing reads, links, and ground-truth updates."""

    def __init__(self, rows_by_key):
        self.rows_by_key = rows_by_key
        self.gt_updates = []
        self.linked = []

    def batch_get_inference_results(self, cache_keys):
        return {
            key: dict(self.rows_by_key[key])
            for key in cache_keys
            if key in self.rows_by_key
        }

    def batch_update_ground_truth(self, updates):
        self.gt_updates.extend(updates)
        return True

    def batch_link_run_samples(self, job_id, cache_keys):
        self.linked.extend(cache_keys)
        return True

    def get_run_metadata(self, config_hash):
        return None

    def save_run_metadata(self, metadata):
        return True

    def close_connections(self):
        pass


def _manager_with_rows(monkeypatch, tmp_path, model_meta, row_template):
    monkeypatch.setenv("KARMA_CACHE_PATH", str(tmp_path / "test.duckdb"))
    monkeypatch.setenv("KARMA_CACHE_TYPE", "duckdb")
    monkeypatch.delenv("KARMA_CONFIG_HASH", raising=False)
    manager = CacheManager(dataset_name="demo/dataset", model_config=model_meta)
    samples = [
        {"input": f"prompt-{i}", "expected_output": row_template[i]["expected"]}
        for i in range(len(row_template))
    ]
    _, cache_keys = manager._batch_generate_cache_keys(samples)
    rows = {}
    for i, spec in enumerate(row_template):
        if spec["row"] is not None:
            rows[cache_keys[i]] = spec["row"]
    fake = _FakeCacheIO(rows)
    manager.cache_io = fake
    manager.job_id = "job-1"
    return manager, fake, samples, cache_keys


def test_failed_row_reuse_rules(tmp_path, monkeypatch, model_meta):
    template = [
        # blank stored response + parse error -> reusable hit
        {
            "expected": "gt",
            "row": {
                "success": False,
                "error_type": "post_processing_error",
                "raw_output": "",
                "ground_truth_output": "gt",
            },
        },
        # legacy row, raw never stored -> miss
        {
            "expected": "gt",
            "row": {
                "success": False,
                "error_type": "post_processing_error",
                "raw_output": None,
                "ground_truth_output": "gt",
            },
        },
        # model error -> miss even with raw stored
        {
            "expected": "gt",
            "row": {
                "success": False,
                "error_type": "model_error",
                "raw_output": "",
                "ground_truth_output": "gt",
            },
        },
        # plain success -> hit
        {
            "expected": "gt",
            "row": {
                "success": True,
                "error_type": None,
                "raw_output": "hello",
                "ground_truth_output": "gt",
            },
        },
    ]
    manager, fake, samples, _ = _manager_with_rows(
        monkeypatch, tmp_path, model_meta, template
    )

    results = manager.batch_fetch_rows(samples)

    assert results[0] is not None, "stored blank response must be reusable"
    assert results[1] is None, "legacy NULL-raw row must stay a miss"
    assert results[2] is None, "model_error row must stay a miss"
    assert results[3] is not None
    assert manager.database_hits == 2
    assert manager.database_misses == 2
    assert fake.gt_updates == []


def test_ground_truth_refresh_on_hit(tmp_path, monkeypatch, model_meta):
    template = [
        # ground truth drifted -> row refreshed
        {
            "expected": "corrected transcript",
            "row": {
                "success": True,
                "error_type": None,
                "raw_output": "hello",
                "ground_truth_output": "old transcript",
            },
        },
        # unchanged ground truth -> untouched
        {
            "expected": "same",
            "row": {
                "success": True,
                "error_type": None,
                "raw_output": "hello",
                "ground_truth_output": "same",
            },
        },
        # miss -> never considered for refresh
        {"expected": "whatever", "row": None},
    ]
    manager, fake, samples, cache_keys = _manager_with_rows(
        monkeypatch, tmp_path, model_meta, template
    )

    results = manager.batch_fetch_rows(samples)

    assert fake.gt_updates == [(cache_keys[0], "corrected transcript")]
    assert results[0]["ground_truth_output"] == "corrected transcript"
    assert results[1]["ground_truth_output"] == "same"
    assert results[2] is None
