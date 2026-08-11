"""Golden hashes — frozen Hash-v2 format (MD5, sort_keys, loader_kwargs-only).

These pin exact hex values. A failure here means the hash FORMAT changed and
every existing cache key would be orphaned. If the change is intentional
(a planned cache-breaking release + re-ETL), update the constants; otherwise
you are about to silently burn the cache — fix the code instead.
"""

from karma.cache.cache_manager import CacheManager
from karma.data_models.dataloader_iterable import (
    Conversation,
    ConversationTurn,
    DataLoaderIterable,
)
from karma.data_models.model_meta import ModelMeta, ModelType


def golden_meta(**overrides) -> ModelMeta:
    kwargs = dict(
        name="golden/model",
        loader_class="karma.models.base_model_abs.BaseModel",
        loader_kwargs={"temperature": 0.5, "max_tokens": 64},
        model_type=ModelType.TEXT_GENERATION,
    )
    kwargs.update(overrides)
    return ModelMeta(**kwargs)


GOLDEN_SAMPLE = dict(
    input="What is 2+2?",
    system_prompt="You are helpful.",
    language="hi",
    other_args={"quality": "high"},
    expected_output="4",
)


def test_golden_config_hash():
    assert (
        CacheManager.hash_for_run_config(
            golden_meta(), dataset_name="golden/ds", dataset_subset="hi", dataset_split="test"
        )
        == "11aabe99788d0322a080f2c2d87c3191"
    )


def test_golden_config_hash_has_no_max_samples_key():
    data = CacheManager.config_data_for_run_hash(
        golden_meta(),
        dataset_name="golden/ds",
        dataset_subset="hi",
        dataset_split="test",
    )
    assert "max_samples" not in data


def test_golden_config_hash_ignores_revision():
    pinned = CacheManager.hash_for_run_config(
        golden_meta(revision="abc123def"),
        dataset_name="golden/ds",
        dataset_subset="hi",
        dataset_split="test",
    )
    unpinned = CacheManager.hash_for_run_config(
        golden_meta(),
        dataset_name="golden/ds",
        dataset_subset="hi",
        dataset_split="test",
    )
    assert pinned == unpinned


def test_golden_sample_hash():
    payload = CacheManager.build_sample_payload(DataLoaderIterable(**GOLDEN_SAMPLE))
    assert CacheManager.generate_hash(payload) == "f586f22513ce96c7a006a2395ee7d62a"


def test_golden_conversation_hash():
    sample = DataLoaderIterable(
        conversation=Conversation(
            conversation_turns=[ConversationTurn(content="hello", role="user")]
        )
    )
    payload = CacheManager.build_sample_payload(sample)
    assert CacheManager.generate_hash(payload) == "010ac7dd178cbd0919c92517a10bc088"


def test_golden_cache_key():
    model_payload = CacheManager.build_model_payload(golden_meta())
    input_hash = CacheManager.generate_hash(
        CacheManager.build_sample_payload(DataLoaderIterable(**GOLDEN_SAMPLE))
    )
    cache_key = CacheManager.generate_hash({**model_payload, "input_hash": input_hash})
    assert cache_key == "b3744e2a50ddda20ecaf76acb9423cbd"
