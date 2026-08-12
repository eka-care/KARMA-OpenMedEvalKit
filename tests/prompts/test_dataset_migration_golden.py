"""Golden checks: migrated datasets resolve their default prompt from
the registry, wiring intact, and the body text does not drift (hash-pinned).

Byte-identity vs the pre-migration constant is verified once, live, at migration
time; these hashes freeze the result so an accidental edit fails CI."""

import hashlib

import pytest

from karma.prompts import resolve_prompt
from karma.registries.dataset_registry import dataset_registry


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# prompt name -> md5 of the exact body carried over from the old dataset constant.
PINNED_BODIES = {
    "medmcqa_cot": "909eb3196e65b12077f494aaef70d1cd",
    "medmcqa_indic_cot": "909eb3196e65b12077f494aaef70d1cd",
    "medqa_cot": "909eb3196e65b12077f494aaef70d1cd",
    "mmlu_medical_cot": "909eb3196e65b12077f494aaef70d1cd",
    "pubmedmcqa_cot": "4cee167ca7ba60064de2093d073c5050",
    "nfimcqa_cot": "3ef8eba1755a9abe950e21e085bd9957",
    "eka_mmlu_medical_cot": "330ce238a1127a2d574a2c932b0ed5ff",
    "indian_drug_mcqa_cot": "f74ce6bafdefd8fa3c87114ec9f2cd81",
    "medxpertqa_cot": "8c18c35f7b130d5590bef228d8a8b8a4",
    "medxpertqa_text_cot": "8c18c35f7b130d5590bef228d8a8b8a4",
    "eka_nfi_cot": "209e4f916f27d7acad366892f6fafd17",
    "slake_cot": "b21281626919a4e60eaf00b93f7de8de",
    "vqa_rad_cot": "b066972045332dc7e45dc788120ee065",
    # system prompts (batch b)
    "healthbench_system": "9d50bd36020623d4dda57655140af181",
    "medical_history_summary_system": "db14dce168fa25b86155f07fe99bd737",
    "medical_records_parsing_system": "e2c78baab75abdfafea024ecf48a7b6d",
    # batch (c) edge cases
    "medqa_usmle_cot": "330ce238a1127a2d574a2c932b0ed5ff",
    "in22conv_translate": "98a65fbd1dfbb17770641d1d52076786",
    "afrimedqa_consumer": "11b409fe2678e88cbda521f9ae321e44",
    "afrimedqa_mcq": "58727511fbd337c15ae5fce2ceac7c9f",
    "afrimedqa_harm_eval": "595a480e77c9214400e8a3bcb967d7f4",
    "amqa_bias_cot": "85f9e6ba52aaee53ea94b23ab90e0709",
    "protocol_retrieval_system": "f3c4402cdda680acbebfbd562fcdcfc8",
    "indian_protocols_base": "cfd71c3e3e329aa025f78f268f24bb08",
    "indian_protocols_tool": "b721491cb7b9a8ac686fb2e70fb99b3d",
    # ASR prompt unification
    "asr_transcribe": "bb4bfd4ea1d530941181a625fe8d4a41",
}


@pytest.mark.parametrize("name", sorted(PINNED_BODIES))
def test_prompt_body_hash_pinned(name):
    assert _md5(resolve_prompt(name).body) == PINNED_BODIES[name]


def test_every_registered_default_prompt_resolves_and_matches_pin():
    """Each dataset that declares default_args.user_prompt must point at a
    registered prompt whose body matches the pinned hash."""
    dataset_registry.discover_datasets(use_cache=False)
    wired = 0
    for info in dataset_registry.datasets.values():
        default_args = info.get("default_args") or {}
        for slot in ("user_prompt", "system_prompt"):
            name = default_args.get(slot)
            if not name:
                continue
            wired += 1
            assert name in PINNED_BODIES, f"unpinned prompt {name!r}"
            assert _md5(resolve_prompt(name).body) == PINNED_BODIES[name]
    assert wired >= 21


def test_medmcqa_resolves_default_prompt_via_registry(monkeypatch):
    from karma.eval_datasets.medmcqa_dataset import MedMCQADataset

    monkeypatch.setattr(
        MedMCQADataset,
        "load_eval_dataset",
        lambda self, *a, **k: [
            {"question": "Q", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "cop": 0}
        ],
    )
    ds = dataset_registry.create_dataset(
        "openlifescienceai/medmcqa", validate_args=True
    )
    assert ds._prompt_provenance["user_prompt"] == "medmcqa_cot@1"
    assert ds.confinement_instructions == resolve_prompt("medmcqa_cot").body
    item = next(iter(ds))
    assert "<QUESTION>" not in item.input and "Q" in item.input
    assert item.system_prompt is None
