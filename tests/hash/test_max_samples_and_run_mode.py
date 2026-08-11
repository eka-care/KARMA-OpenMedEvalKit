"""max_samples is a job column, never hashed; dev/prod derivation."""

from karma.cache.cache_manager import CacheManager
from karma.data_models.model_meta import ModelMeta, ModelType
from karma.utils.run_mode import derive_run_mode


def make_meta() -> ModelMeta:
    return ModelMeta(
        name="test-model",
        loader_class="karma.models.base_model_abs.BaseModel",
        loader_kwargs={"temperature": 0.0},
        model_type=ModelType.TEXT_GENERATION,
    )


DATASET = dict(dataset_name="ds", dataset_subset="en", dataset_split="test")


def test_config_hash_ignores_max_samples():
    data = CacheManager.config_data_for_run_hash(make_meta(), **DATASET)
    assert "max_samples" not in data


def test_cache_key_payload_has_no_max_samples():
    payload = CacheManager.build_model_payload(make_meta())
    assert "max_samples" not in payload


def test_run_mode_dev_when_max_samples_set(monkeypatch):
    monkeypatch.setattr("karma.utils.run_mode._code_matches_main_head", lambda: True)
    assert derive_run_mode(max_samples=50) == "dev"


def test_run_mode_prod_when_full_run_on_main_head(monkeypatch):
    monkeypatch.setattr("karma.utils.run_mode._code_matches_main_head", lambda: True)
    assert derive_run_mode(max_samples=None) == "prod"


def test_run_mode_dev_when_code_off_main_head(monkeypatch):
    monkeypatch.setattr("karma.utils.run_mode._code_matches_main_head", lambda: False)
    assert derive_run_mode(max_samples=None) == "dev"
