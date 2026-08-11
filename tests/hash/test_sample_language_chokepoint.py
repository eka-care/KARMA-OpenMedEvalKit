"""Language chokepoint — benchmark standardizes sample.language before hash+model."""

from types import SimpleNamespace

import pytest

from karma.benchmark import Benchmark
from karma.data_models.dataloader_iterable import DataLoaderIterable


def make_bench(config_language):
    bench = object.__new__(Benchmark)
    bench.dataset = SimpleNamespace(language=config_language)
    bench.logger = SimpleNamespace(warning=lambda *a, **k: None)
    return bench


def test_config_language_forces_row_value():
    bench = make_bench("hi")
    s = DataLoaderIterable(input="x", language="unknown")
    bench._standardize_sample_languages([s])
    assert s.language == "hi"


def test_multilingual_row_value_normalized():
    bench = make_bench(None)
    s = DataLoaderIterable(input="x", language="hindi")
    bench._standardize_sample_languages([s])
    assert s.language == "hi"


def test_multilingual_unmapped_row_value_fails_eval():
    bench = make_bench(None)
    s = DataLoaderIterable(input="x", language="xx-weird")
    with pytest.raises(ValueError):
        bench._standardize_sample_languages([s])


def test_config_language_covers_none_language_rows():
    bench = make_bench("hi")
    s = DataLoaderIterable(input="x", other_args={"quality": "high"})
    bench._standardize_sample_languages([s])
    assert s.language == "hi"


def test_multilingual_none_language_row_untouched():
    bench = make_bench(None)
    s = DataLoaderIterable(input="x")
    bench._standardize_sample_languages([s])
    assert s.language is None
