"""HF revision resolution + hash effect."""

from unittest.mock import MagicMock, patch

from karma.cache.cache_manager import CacheManager
from karma.data_models.model_meta import ModelMeta, ModelType
from karma.utils.hf import looks_like_hub_id, resolve_hf_revision


def make_meta(**overrides) -> ModelMeta:
    kwargs = dict(
        name="org/model",
        loader_class="karma.models.base_model_abs.BaseModel",
        loader_kwargs={"temperature": 0.0},
        model_type=ModelType.TEXT_GENERATION,
    )
    kwargs.update(overrides)
    return ModelMeta(**kwargs)


def test_hub_id_detection():
    assert looks_like_hub_id("Qwen/Qwen3-0.6B")
    assert looks_like_hub_id("google/medgemma-4b-it")
    assert not looks_like_hub_id("gpt-4o")
    assert not looks_like_hub_id("us.anthropic.claude-sonnet-4")
    assert not looks_like_hub_id(None)
    assert not looks_like_hub_id("/tmp")
    assert not looks_like_hub_id("a/b/c")


def test_resolve_skips_non_hub_ids():
    assert resolve_hf_revision("gpt-4o") is None


def test_resolve_returns_sha_from_api():
    resolve_hf_revision.cache_clear()
    api = MagicMock()
    api.model_info.return_value.sha = "deadbeef"
    with patch("huggingface_hub.HfApi", return_value=api):
        assert resolve_hf_revision("org/some-model") == "deadbeef"


def test_resolve_none_on_api_failure():
    resolve_hf_revision.cache_clear()
    with patch("huggingface_hub.HfApi", side_effect=RuntimeError("offline")):
        assert resolve_hf_revision("org/other-model") is None


def test_revision_never_changes_config_hash():
    ds = dict(dataset_name="ds", dataset_subset=None, dataset_split=None)
    a = CacheManager.hash_for_run_config(make_meta(revision="sha1"), **ds)
    b = CacheManager.hash_for_run_config(make_meta(revision="sha2"), **ds)
    c = CacheManager.hash_for_run_config(make_meta(revision=None), **ds)
    assert a == b == c
