"""Prompt registry: Prompt validation, register/resolve."""

import pytest

import karma.prompts as prompts_mod
from karma.prompts import Prompt, PromptNotFoundError, resolve_prompt, _register


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = dict(prompts_mod._REGISTRY)
    saved_flag = prompts_mod._discovered
    prompts_mod._REGISTRY.clear()
    prompts_mod._discovered = True  # skip disk discovery; tests register directly
    yield
    prompts_mod._REGISTRY.clear()
    prompts_mod._REGISTRY.update(saved)
    prompts_mod._discovered = saved_flag


def test_role_must_be_valid():
    with pytest.raises(ValueError):
        Prompt(name="x", version=1, role="judge", body="b")


def test_register_and_resolve():
    p = Prompt(name="medmcqa_cot", version=1, role="user", body="Answer. <QUESTION>")
    _register(p)
    assert resolve_prompt("medmcqa_cot") is p
    assert resolve_prompt("medmcqa_cot").body == "Answer. <QUESTION>"


def test_unknown_name_raises():
    with pytest.raises(PromptNotFoundError):
        resolve_prompt("does_not_exist")


def test_duplicate_name_conflict_raises():
    _register(Prompt(name="dup", version=1, role="user", body="a"))
    with pytest.raises(ValueError):
        _register(Prompt(name="dup", version=1, role="user", body="DIFFERENT"))


def test_reregister_identical_is_idempotent():
    _register(Prompt(name="same", version=2, role="system", body="persona"))
    _register(Prompt(name="same", version=2, role="system", body="persona"))
    assert resolve_prompt("same").version == 2
