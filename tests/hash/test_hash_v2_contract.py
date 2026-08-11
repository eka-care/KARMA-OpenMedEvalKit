"""Hash v2 contract — behavioral invariants for the cache-key/config-hash format.

These are property tests: they assert what the hash MUST and MUST NOT depend
on, not exact hex values (those are frozen separately in the golden tests once
the format is final). They encode the Hash v2 decisions:

- model payload = ``loader_kwargs`` only (+ injected model_name_or_path);
  documentation-only ModelMeta fields never touch a hash. Revision is
  provenance-only — a docs-only hub commit must not burn the cache — and
  loader_class is identity-exempt too (models spec M6): neither enters
  any hash.
- infra keys in ``loader_kwargs`` are a registration error (denylist
  validates, never strips).
- sample payload is a whitelist: generation inputs only, as typed root
  fields (``language`` included). Ground truth, eval-side rubric data, and
  ALL of ``other_args`` never touch a hash.
- dict key order never changes a hash (``sort_keys``).

Run: ``pytest tests/hash/``.
"""

import pytest

from karma.cache.cache_manager import CacheManager
from karma.data_models.dataloader_iterable import (
    Conversation,
    ConversationTurn,
    DataLoaderIterable,
)
from karma.data_models.model_meta import ModelMeta, ModelType


def make_meta(**overrides) -> ModelMeta:
    """A minimal, valid ModelMeta with sensible defaults for hashing tests."""
    kwargs = dict(
        name="test-model",
        loader_class="karma.models.base_model_abs.BaseModel",
        loader_kwargs={"temperature": 0.0, "max_tokens": 128},
        model_type=ModelType.TEXT_GENERATION,
    )
    kwargs.update(overrides)
    return ModelMeta(**kwargs)


DATASET = dict(dataset_name="ds", dataset_subset="en", dataset_split="test")


def config_hash(meta: ModelMeta) -> str:
    return CacheManager.hash_for_run_config(meta, **DATASET)


# --------------------------------------------------------------------------- #
# generate_hash: order independence (sort_keys)
# --------------------------------------------------------------------------- #


def test_generate_hash_is_key_order_independent():
    a = CacheManager.generate_hash({"a": 1, "b": 2, "c": 3})
    b = CacheManager.generate_hash({"c": 3, "b": 2, "a": 1})
    assert a == b


def test_generate_hash_nested_order_independent():
    a = CacheManager.generate_hash({"outer": {"x": 1, "y": 2}})
    b = CacheManager.generate_hash({"outer": {"y": 2, "x": 1}})
    assert a == b


# --------------------------------------------------------------------------- #
# build_model_payload: loader_kwargs only + injection + denylist
# --------------------------------------------------------------------------- #


def test_model_payload_injects_name_as_path_when_model_path_unset():
    payload = CacheManager.build_model_payload(make_meta(name="Qwen/Qwen3-0.6B"))
    assert payload["model_name_or_path"] == "Qwen/Qwen3-0.6B"


def test_model_payload_prefers_model_path_over_name():
    payload = CacheManager.build_model_payload(
        make_meta(name="alias", model_path="/local/path")
    )
    assert payload["model_name_or_path"] == "/local/path"


def test_model_payload_never_contains_revision():
    payload = CacheManager.build_model_payload(make_meta(revision="abc123"))
    assert "revision" not in payload


def test_model_payload_never_contains_loader_class():
    a = CacheManager.build_model_payload(make_meta())
    b = CacheManager.build_model_payload(
        make_meta(loader_class="karma.cache.cache_manager.CacheManager")
    )
    assert "loader_class" not in a
    assert CacheManager.generate_hash(a) == CacheManager.generate_hash(b)


def test_model_payload_preserves_generation_kwargs():
    payload = CacheManager.build_model_payload(
        make_meta(loader_kwargs={"temperature": 0.7, "top_p": 0.9})
    )
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.9


def test_model_payload_excludes_documentation_fields():
    payload = CacheManager.build_model_payload(
        make_meta(description="a long description", reference="http://x")
    )
    assert "description" not in payload
    assert "reference" not in payload
    assert "modalities" not in payload
    assert "languages" not in payload
    assert "default_eval_kwargs" not in payload


@pytest.mark.parametrize("infra_key", ["device", "timeout", "max_retries", "max_workers"])
def test_model_payload_rejects_infra_keys(infra_key):
    meta = make_meta(loader_kwargs={"temperature": 0.0, infra_key: "whatever"})
    with pytest.raises(ValueError):
        CacheManager.build_model_payload(meta)


# --------------------------------------------------------------------------- #
# config_hash: driven by loader_kwargs, immune to documentation noise
# --------------------------------------------------------------------------- #


def test_config_hash_ignores_description():
    assert config_hash(make_meta(description="one")) == config_hash(
        make_meta(description="two")
    )


def test_config_hash_ignores_reference_and_release_date():
    a = make_meta(reference="http://a", release_date="2024-01-01")
    b = make_meta(reference="http://b", release_date="2025-12-31")
    assert config_hash(a) == config_hash(b)


def test_config_hash_changes_with_generation_kwargs():
    a = make_meta(loader_kwargs={"temperature": 0.0})
    b = make_meta(loader_kwargs={"temperature": 0.7})
    assert config_hash(a) != config_hash(b)


def test_config_hash_changes_with_model_name():
    assert config_hash(make_meta(name="model-a")) != config_hash(
        make_meta(name="model-b")
    )


def test_config_hash_changes_with_dataset():
    meta = make_meta()
    a = CacheManager.hash_for_run_config(meta, dataset_name="ds1", dataset_subset="en", dataset_split="test")
    b = CacheManager.hash_for_run_config(meta, dataset_name="ds2", dataset_subset="en", dataset_split="test")
    assert a != b


# --------------------------------------------------------------------------- #
# build_sample_payload: generation-input whitelist
# --------------------------------------------------------------------------- #


def sample_hash(sample: DataLoaderIterable) -> str:
    return CacheManager.generate_hash(CacheManager.build_sample_payload(sample))


def test_sample_payload_includes_generation_inputs():
    payload = CacheManager.build_sample_payload(
        DataLoaderIterable(
            input="q",
            system_prompt="sys",
            language="en",
        )
    )
    assert payload.get("input") == "q"
    assert payload.get("system_prompt") == "sys"
    assert payload.get("language") == "en"


def test_sample_payload_excludes_expected_output():
    payload = CacheManager.build_sample_payload(
        DataLoaderIterable(input="q", expected_output="the answer")
    )
    assert "expected_output" not in str(payload)


def test_sample_payload_excludes_rubric():
    payload = CacheManager.build_sample_payload(
        DataLoaderIterable(input="q", rubric_to_evaluate=[])
    )
    assert "rubric_to_evaluate" not in payload


def test_cache_key_ignores_expected_output():
    a = DataLoaderIterable(input="q", expected_output="answer-A")
    b = DataLoaderIterable(input="q", expected_output="answer-B")
    assert sample_hash(a) == sample_hash(b)


def test_cache_key_ignores_other_args_entirely():
    a = DataLoaderIterable(input="q", other_args={"quality": "high", "language": "en"})
    b = DataLoaderIterable(input="q", other_args={"quality": "low", "language": "hi"})
    assert sample_hash(a) == sample_hash(b)


def test_cache_key_changes_with_language_field():
    a = DataLoaderIterable(input="q", language="en")
    b = DataLoaderIterable(input="q", language="hi")
    assert sample_hash(a) != sample_hash(b)


def test_cache_key_changes_with_input():
    a = DataLoaderIterable(input="question one")
    b = DataLoaderIterable(input="question two")
    assert sample_hash(a) != sample_hash(b)


def test_cache_key_changes_with_system_prompt():
    a = DataLoaderIterable(input="q", system_prompt="prompt one")
    b = DataLoaderIterable(input="q", system_prompt="prompt two")
    assert sample_hash(a) != sample_hash(b)


def test_cache_key_changes_with_conversation():
    a = DataLoaderIterable(
        conversation=Conversation(
            conversation_turns=[ConversationTurn(content="hi", role="user")]
        )
    )
    b = DataLoaderIterable(
        conversation=Conversation(
            conversation_turns=[ConversationTurn(content="bye", role="user")]
        )
    )
    assert sample_hash(a) != sample_hash(b)
