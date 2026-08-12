import json

from karma.metrics.rubrics.rubric_evaluation import RubricMetric
from karma.data_models.dataloader_iterable import RubricCriteria


class _DummyModel:
    """Test double for OpenAILLM that returns preset responses."""

    def __init__(self, responses):
        self._responses = responses
        self.call_index = 0

    def run(self, inputs):
        response = self._responses[self.call_index]
        self.call_index += 1
        return [response for _ in inputs]


def test_rubric_metric_falls_back_for_malformed_batch(monkeypatch):
    malformed_batch_response = json.dumps([{"unexpected": True}])
    fallback_response_one = json.dumps(
        {"criteria_met": True, "explanation": "first rubric"}
    )
    fallback_response_two = json.dumps(
        {"criteria_met": False, "explanation": "second rubric"}
    )

    dummy_model_instances = []

    def _dummy_factory(*args, **kwargs):
        model = _DummyModel(
            [malformed_batch_response, fallback_response_one, fallback_response_two]
        )
        dummy_model_instances.append(model)
        return model

    monkeypatch.setattr(
        "karma.metrics.rubrics.rubric_evaluation.OpenAILLM", _dummy_factory
    )

    metric = RubricMetric(
        metric_name="rubric_evaluation",
        provider_to_use="openai",
        model_id="dummy",
    )

    rubric_batch = [
        RubricCriteria(criterion="c1", points=1, tags=[]),
        RubricCriteria(criterion="c2", points=2, tags=[]),
    ]

    results, raw_response = metric._evaluate_batch("{}", rubric_batch)

    assert [item["criteria_met"] for item in results] == [True, False]
    assert [item["rubric"] for item in results] == rubric_batch
    # The raw judge response is returned for the rubric_outputs cache.
    assert raw_response == malformed_batch_response

    # Ensure the malformed batch triggered the fallback path (1 batch + 2 fallback calls)
    assert dummy_model_instances[0].call_index == 3
