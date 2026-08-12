"""Base-class prompt resolution + central system-prompt stamp."""

import pytest

import karma.prompts as prompts_mod
from karma.prompts import Prompt, PromptNotFoundError
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset


@pytest.fixture(autouse=True)
def _registry():
    saved, flag = dict(prompts_mod._REGISTRY), prompts_mod._discovered
    prompts_mod._REGISTRY.clear()
    prompts_mod._discovered = True
    prompts_mod._register(Prompt(name="mcq_task", version=3, role="user", body="Answer. <QUESTION>"))
    prompts_mod._register(Prompt(name="persona", version=1, role="system", body="You are a doctor."))
    yield
    prompts_mod._REGISTRY.clear()
    prompts_mod._REGISTRY.update(saved)
    prompts_mod._discovered = flag


class _Dataset(BaseMultimodalDataset):
    """Minimal concrete dataset: no HF load, echoes input; optionally sets system_prompt."""

    def __init__(self, set_own_system=False, **kwargs):
        self._set_own_system = set_own_system
        super().__init__(dataset_name="ds", split="test", **kwargs)

    def load_eval_dataset(self, *a, **k):
        return [{"q": "what?"}]

    def format_item(self, sample):
        return DataLoaderIterable(
            input=self.confinement_instructions.replace("<QUESTION>", sample["q"]),
            system_prompt="own" if self._set_own_system else None,
        )


def _first(ds):
    return next(iter(ds))


def test_user_prompt_name_resolves_from_registry():
    ds = _Dataset(user_prompt="mcq_task")
    assert ds.confinement_instructions == "Answer. <QUESTION>"
    assert ds._prompt_provenance["user_prompt"] == "mcq_task@3"
    assert _first(ds).input == "Answer. what?"


def test_user_prompt_text_is_raw_and_beats_name():
    ds = _Dataset(user_prompt="mcq_task", user_prompt_text="RAW <QUESTION>")
    assert ds.confinement_instructions == "RAW <QUESTION>"
    assert ds._prompt_provenance["user_prompt"] == "inline"


def test_unknown_user_prompt_name_errors():
    with pytest.raises(PromptNotFoundError):
        _Dataset(user_prompt="nope")


def test_no_prompt_defaults_are_inert():
    ds = _Dataset()
    assert ds.confinement_instructions == ""
    assert ds.system_prompt is None
    assert ds._prompt_provenance == {"user_prompt": None, "system_prompt": None}
    assert _first(ds).system_prompt is None


def test_system_prompt_name_stamped_centrally():
    ds = _Dataset(system_prompt="persona")
    assert ds.system_prompt == "You are a doctor."
    assert ds._prompt_provenance["system_prompt"] == "persona@1"
    assert _first(ds).system_prompt == "You are a doctor."


def test_system_prompt_text_raw_stamped():
    ds = _Dataset(system_prompt_text="RAW PERSONA")
    assert _first(ds).system_prompt == "RAW PERSONA"
    assert ds._prompt_provenance["system_prompt"] == "inline"


def test_central_stamp_never_overrides_item_set_system_prompt():
    ds = _Dataset(system_prompt="persona", set_own_system=True)
    # format_item already set system_prompt="own" -> stamp must not clobber it.
    assert _first(ds).system_prompt == "own"


class _AudioDataset(BaseMultimodalDataset):
    """format_item sets audio only (no input) — like an ASR dataset."""

    def __init__(self, **kwargs):
        super().__init__(dataset_name="ds", split="test", **kwargs)

    def load_eval_dataset(self, *a, **k):
        return [{"x": 1}]

    def format_item(self, sample):
        return DataLoaderIterable(audio=b"AUDIO")


def test_user_prompt_stamped_as_input_when_format_item_omits_it():
    ds = _AudioDataset(user_prompt="mcq_task")  # verbatim instruction, no <QUESTION> use
    item = _first(ds)
    assert item.input == "Answer. <QUESTION>"  # body of mcq_task fixture
    assert item.audio == b"AUDIO"


def test_no_user_prompt_leaves_input_none():
    assert _first(_AudioDataset()).input is None


def test_user_prompt_stamp_never_overrides_existing_input():
    ds = _Dataset(user_prompt="mcq_task")  # _Dataset.format_item already sets input
    assert _first(ds).input == "Answer. what?"  # substituted, not the raw body
