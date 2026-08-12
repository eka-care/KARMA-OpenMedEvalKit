import uuid
from types import SimpleNamespace

import pytest

import karma.cache.cache_manager as cache_manager_module
from karma.benchmark import Benchmark
from karma.cache import CacheManager
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.data_models.model_meta import ModelMeta, ModelType

FORCED_JOB_ID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture()
def model_meta():
    return ModelMeta(
        name="test-model",
        loader_class="karma.models.base_model_abs.BaseModel",
        loader_kwargs={},
        default_eval_kwargs={},
        model_type=ModelType.TEXT_GENERATION,
    )


@pytest.fixture()
def duckdb_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KARMA_CACHE_PATH", str(tmp_path / "cache.duckdb"))
    monkeypatch.setenv("KARMA_CACHE_TYPE", "duckdb")
    monkeypatch.delenv("KARMA_CONFIG_HASH", raising=False)
    monkeypatch.setattr(cache_manager_module, "_forced_job_id_claimed", False)


def _make_manager(model_meta, dataset_name="demo/dataset"):
    return CacheManager(dataset_name=dataset_name, model_config=model_meta)


def test_forced_job_id_claimed_once_per_process(duckdb_env, monkeypatch, model_meta):
    monkeypatch.setenv("KARMA_JOB_ID", FORCED_JOB_ID)

    first = _make_manager(model_meta, "dataset/a")
    second = _make_manager(model_meta, "dataset/b")

    assert first.job_id == FORCED_JOB_ID
    assert second.job_id != FORCED_JOB_ID
    uuid.UUID(second.job_id)


def test_no_forced_job_id_mints_uuid(duckdb_env, monkeypatch, model_meta):
    monkeypatch.delenv("KARMA_JOB_ID", raising=False)

    manager = _make_manager(model_meta)

    uuid.UUID(manager.job_id)


def _fetch_with_rows(manager, row_specs):
    samples = [
        {"input": f"question {i}", "other_args": {}} for i in range(len(row_specs))
    ]

    def fake_batch_get(cache_keys):
        return {
            key: dict(spec, cache_key=key)
            for key, spec in zip(cache_keys, row_specs)
            if spec is not None
        }

    manager.cache_io.batch_get_inference_results = fake_batch_get
    return manager.batch_fetch_rows(samples)


def test_fetch_reuses_extraction_failures_with_raw(duckdb_env, model_meta):
    manager = _make_manager(model_meta)
    rows = _fetch_with_rows(
        manager,
        [
            {"success": True, "raw_output": "B", "error_type": None},
            {"success": False, "raw_output": "The answer is B", "error_type": None},
            {
                "success": False,
                "raw_output": "The answer is B",
                "error_type": "post_processing_error",
            },
        ],
    )

    assert all(row is not None for row in rows)


def test_latest_job_prefers_prod_over_newer_dev(duckdb_env, model_meta):
    manager = _make_manager(model_meta)
    io = manager.cache_io
    io.save_run_metadata(
        {
            "job_id": "11111111-1111-1111-1111-11111111aaaa",
            "config_hash": manager.config_hash,
            "run_mode": "prod",
        }
    )
    io.save_run_metadata(
        {
            "job_id": "11111111-1111-1111-1111-11111111bbbb",
            "config_hash": manager.config_hash,
            "run_mode": "dev",
        }
    )

    assert (
        io._latest_job_for_config(manager.config_hash)
        == "11111111-1111-1111-1111-11111111aaaa"
    )


def test_update_run_counts_five_way_split(duckdb_env, model_meta):
    manager = _make_manager(model_meta)
    io = manager.cache_io
    io.update_run_counts(
        manager.config_hash,
        total=10,
        processed=7,
        error=2,
        job_id=manager.job_id,
        post_processing_errors=1,
        cache_hits=5,
        duration_seconds=12.5,
    )
    row = io.db_io.fetchone(
        "SELECT total_samples, processed_samples, error_count, "
        "post_processing_error_count, cache_hits, duration_seconds "
        f"FROM parrotlet_omni_benchmark_cache_run_metadata WHERE job_id = ?",
        [manager.job_id],
    )
    assert tuple(row) == (10, 7, 2, 1, 5, 12.5)


def test_fetch_misses_model_errors_and_null_raw(duckdb_env, model_meta):
    manager = _make_manager(model_meta)
    rows = _fetch_with_rows(
        manager,
        [
            {"success": False, "raw_output": "partial junk", "error_type": "model_error"},
            # "" is a stored blank response — reusable; NULL is legacy — miss.
            {"success": False, "raw_output": "", "error_type": None},
            {"success": False, "raw_output": None, "error_type": None},
            None,
        ],
    )

    assert rows[0] is None
    assert rows[1] is not None
    assert rows[2] is None
    assert rows[3] is None
    assert manager.database_misses == 3
    assert manager.database_hits == 1


def _make_bench(cached_rows, extract):
    bench = object.__new__(Benchmark)
    bench.refresh_cache = False
    bench.verbose_mode = False
    bench.logger = SimpleNamespace(info=lambda *a, **k: None)
    bench.dataset = SimpleNamespace(extract_prediction=extract)
    bench.cache_manager = SimpleNamespace(batch_fetch_rows=lambda samples: cached_rows)
    return bench


def test_reextraction_success_clears_stale_error_type():
    row = {
        "raw_output": "The answer is B",
        "success": False,
        "error_type": "post_processing_error",
    }
    bench = _make_bench([row], lambda raw, other_args=None: ("B", True))
    results, _, _ = bench.fetch_from_cache([DataLoaderIterable(input="q")])

    assert results[0]["success"] is True
    assert results[0]["error_type"] is None


def test_reextraction_failure_marks_post_processing_error():
    row = {"raw_output": "garbled", "success": True, "error_type": None}
    bench = _make_bench([row], lambda raw, other_args=None: ("", False))
    results, _, _ = bench.fetch_from_cache([DataLoaderIterable(input="q")])

    assert results[0]["success"] is False
    assert results[0]["error_type"] == "post_processing_error"
