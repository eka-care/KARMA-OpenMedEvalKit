"""Language normalization."""

import pytest

from karma.utils.language import metric_language, normalize_language, sample_language


def test_iso_codes_pass_through():
    assert normalize_language("hi") == "hi"
    assert normalize_language("en") == "en"
    assert normalize_language("ta") == "ta"


def test_names_map_to_iso():
    assert normalize_language("hindi") == "hi"
    assert normalize_language("english") == "en"
    assert normalize_language("Tamil") == "ta"
    assert normalize_language("MALAYALAM") == "ml"


def test_variants_map_to_iso():
    assert normalize_language("en-IN") == "en"
    assert normalize_language("en-US") == "en"


def test_idempotent():
    assert normalize_language(normalize_language("hindi")) == "hi"


def test_none_and_empty():
    assert normalize_language(None) is None
    assert normalize_language("  ") is None


def test_unknown_raises():
    with pytest.raises(ValueError):
        normalize_language("klingon")
    with pytest.raises(ValueError):
        normalize_language("hindii")


def test_sample_language_row_helper():
    assert sample_language("hindi") == "hi"
    assert sample_language("unknown") is None
    assert sample_language("") is None
    assert sample_language(None) is None
    with pytest.raises(ValueError):
        sample_language("klingon")


def test_metric_language_is_lenient():
    assert metric_language("hindi") == "hi"
    assert metric_language(None) == "en"
    assert metric_language("zulu") == "zu"
    assert metric_language("klingon") == "en"
