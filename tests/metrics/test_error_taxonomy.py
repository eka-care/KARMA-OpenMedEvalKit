from types import SimpleNamespace

from karma.benchmark import Benchmark
from karma.data_models.dataloader_iterable import DataLoaderIterable


def _bench(extract):
    bench = object.__new__(Benchmark)
    bench.logger = SimpleNamespace(warning=lambda *a, **k: None)
    bench.dataset = SimpleNamespace(extract_prediction=extract)
    return bench


SAMPLE = DataLoaderIterable(input="q")


def test_clean_extraction():
    bench = _bench(lambda raw, other_args=None: ("B", True))
    assert bench._extract_with_taxonomy("The answer is B", SAMPLE) == ("B", True, None)


def test_extraction_returns_failure():
    bench = _bench(lambda raw, other_args=None: ("", False))
    assert bench._extract_with_taxonomy("garbled", SAMPLE) == (
        "",
        False,
        "post_processing_error",
    )


def test_extraction_exception_survives():
    def boom(raw, other_args=None):
        raise KeyError("malformed")

    bench = _bench(boom)
    assert bench._extract_with_taxonomy("junk", SAMPLE) == (
        "",
        False,
        "post_processing_error",
    )


def test_model_error_wins_over_clean_extraction():
    bench = _bench(lambda raw, other_args=None: ("B", True))
    assert bench._extract_with_taxonomy("B", SAMPLE, model_error=True) == (
        "B",
        False,
        "model_error",
    )


def _item(prediction, success=True, error_type=None):
    return {
        "prediction": prediction,
        "expected_output": "ref",
        "entities": None,
        "quality": None,
        "sample": None,
        "sample_serialized": None,
        "success": success,
        "error_type": error_type,
    }


def test_field_support_explodes_into_metric_records():
    saved = {}

    def evaluate(**kwargs):
        return {
            "semantic_wer_by_quality": {"Quality_high_Semantic_WER": 0.2},
            "overall_wer": 0.25,
            "_overall_value": 0.25,
            "_field_support": {
                "Quality_high_Semantic_WER": {"value": 0.2, "scored": 3, "total": 4}
            },
        }

    bench = object.__new__(Benchmark)
    bench.logger = SimpleNamespace(
        info=lambda *a, **k: None, exception=lambda *a, **k: None
    )
    bench.dataset = SimpleNamespace(
        postprocess=lambda x: x, dataset_name="demo/ds", language=None
    )
    bench.enable_cache = True
    bench.cache_manager = SimpleNamespace(
        save_metric_results=lambda records: saved.setdefault("records", records),
        get_cache_identifiers_for_sample=lambda s: None,
    )
    metric = SimpleNamespace(
        metric_name="semantic_wer_by_quality",
        evaluate=evaluate,
        non_percentage_fields=None,
    )

    bench.compute_metrics([_item("hyp")], metrics=[metric], persist=True)

    records = {r["metric_name"]: r for r in saved["records"]}
    parent = records["semantic_wer_by_quality"]
    field = records["Quality_high_Semantic_WER"]
    assert parent["value"] == 0.25
    assert "_field_support" not in (parent["payload"] or {})
    assert field["value"] == 0.2
    assert field["total_count"] == 4 and field["scored_count"] == 3


def test_rates_with_support_helper():
    from karma.metrics.analysis_metrics import _rates_with_support

    rates = _rates_with_support(1, 2, 3, 10, ["ref a", "", "ref b"])
    assert rates["substitution_rate"] == 0.1
    support = rates["_field_support"]["deletion_rate"]
    assert support == {"value": 0.2, "scored": 2, "total": 3}


def test_compute_metrics_excludes_failed_samples():
    seen = {}

    def evaluate(**kwargs):
        seen["predictions"] = kwargs["predictions"]
        return 1.0

    bench = object.__new__(Benchmark)
    bench.logger = SimpleNamespace(
        info=lambda *a, **k: None, exception=lambda *a, **k: None
    )
    bench.dataset = SimpleNamespace(
        postprocess=lambda x: x, dataset_name="demo/ds", language=None
    )
    bench.enable_cache = False
    bench.cache_manager = None
    metric = SimpleNamespace(metric_name="acc", evaluate=evaluate)

    items = [
        _item("good"),
        _item("", success=False, error_type="model_error"),
        _item("junk", success=False, error_type="post_processing_error"),
        _item("", success=False),  # legacy: no error_type -> model_error bucket
    ]
    scores = bench.compute_metrics(items, metrics=[metric], persist=False)

    assert seen["predictions"] == ["good"]
    assert scores == {"acc": 1.0}
    assert bench.metric_error_counts == {
        "model_error": 2,
        "post_processing_error": 1,
    }
