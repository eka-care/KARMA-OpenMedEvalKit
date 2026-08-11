"""Judge cache (rubric_outputs) for json_rubric_evaluation: hit/miss, persist rules."""

import json
import types

from karma.metrics.rubrics.json_rubric_evaluation import JsonRubricEvaluationMetric
from karma.cache.cache_manager import CacheManager


class _FakeCacheManager:
    """In-memory stand-in backed by a dict rubric_outputs store."""

    def __init__(self):
        self.store = {}

    def get_cache_identifiers_for_sample(self, sample):
        return {"cache_key": "ck_" + str(sample.get("input", "x")), "dataset_row_hash": "rh"}

    def generate_hash(self, data):
        return CacheManager.generate_hash(data)

    def save_rubric_outputs(self, records):
        for record in records:
            self.store[record["judge_hash"]] = record
        return True

    def get_rubric_outputs(self, judge_hashes):
        return {h: self.store[h] for h in judge_hashes if h in self.store}

    def batch_save_rubric_question_results(self, records):
        return True

_OMISSION_AXES = {
    "entity": [
        {"field": "semantic_meaning_accuracy", "passes_when": {"eq": 1}},
    ],
}
_MARKER = "=== RUBRICS TO EVALUATE ==="


class _EchoJudge:
    def __init__(self):
        self.inputs = []

    def run(self, inputs):
        out = []
        for item in inputs:
            self.inputs.append(item.input)
            out.append(json.dumps(self._score(item.input)))
        return out

    @staticmethod
    def _score(text):
        if _MARKER not in text:
            return []
        block = text.split(_MARKER, 1)[1].split("GENERATED OUTPUT:")[0]
        arr = json.loads(block[block.find("["): block.rfind("]") + 1])
        return [
            {"rubric_id": r["rubric_id"], "type": "entity",
             "semantic_meaning_accuracy": 1}
            for r in arr
        ]


class _GarbageJudge(_EchoJudge):
    def run(self, inputs):
        self.inputs.extend(i.input for i in inputs)
        return ["not json {{{" for _ in inputs]


def _metric(monkeypatch, judge, chunk_size=10, reasoning_effort=None):
    monkeypatch.setattr(
        "karma.metrics.rubrics.json_rubric_evaluation.OpenAILLM",
        lambda *a, **k: judge,
    )
    return JsonRubricEvaluationMetric(
        metric_name="json_rubric_evaluation", provider_to_use="openai",
        model_id="dummy", chunk_size=chunk_size, reasoning_effort=reasoning_effort,
    )


def _sample(n_rubrics, prefix="R"):
    rubrics = [
        {"rubric_id": f"{prefix}-{i:03d}", "type": "entity",
         "evaluation_instruction": f"c{i}"}
        for i in range(1, n_rubrics + 1)
    ]
    pass_spec = {
        "name": "omission", "prompt": "INSTR", "id_field": "rubric_id",
        "type_field": "type", "axes": _OMISSION_AXES,
        "metadata": [{"rubric_id": r["rubric_id"]} for r in rubrics],
        "response_array_key": None, "rubrics": rubrics,
    }
    return types.SimpleNamespace(
        other_args={"rubric_passes": [pass_spec], "document_type": "note",
                    "language": "en"}
    )


def _run(metric, cm, predictions, samples):
    return metric.evaluate(
        predictions,
        samples=samples,
        successes=[True] * len(samples),
        serialized_samples=[{"input": f"s{i}"} for i in range(len(samples))],
        cache_manager=cm,
    )


def test_cold_run_populates_cache(monkeypatch):
    cm = _FakeCacheManager()
    judge = _EchoJudge()
    metric = _metric(monkeypatch, judge, chunk_size=10)
    _run(metric, cm, ["GEN"], [_sample(25)])
    assert len(judge.inputs) == 3          # 25 rubrics / 10 -> 3 chunk calls
    assert len(cm.store) == 3
    rows = list(cm.store.values())
    assert all(r["cache_key"] == "ck_s0" for r in rows)
    assert all(json.loads(r["raw_response"]) for r in rows)
    with_criteria = [r for r in rows if r["criteria"]]
    assert len(with_criteria) == 1
    assert len(with_criteria[0]["criteria"]) == 25


def test_warm_run_zero_judge_calls_same_score(monkeypatch):
    cm = _FakeCacheManager()
    cold = _run(_metric(monkeypatch, _EchoJudge()), cm, ["GEN"], [_sample(25)])
    warm_judge = _EchoJudge()
    warm = _run(_metric(monkeypatch, warm_judge), cm, ["GEN"], [_sample(25)])
    assert warm_judge.inputs == []
    assert warm["json_rubric_evaluation"]["omission"]["num_items"] == 25
    assert (
        warm["json_rubric_evaluation"]["omission_rate"]
        == cold["json_rubric_evaluation"]["omission_rate"]
    )


def test_changed_prediction_misses_only_its_jobs(monkeypatch):
    cm = _FakeCacheManager()
    samples = [_sample(5, prefix="A"), _sample(5, prefix="B")]
    _run(_metric(monkeypatch, _EchoJudge()), cm, ["G1", "G2"], samples)
    judge = _EchoJudge()
    _run(_metric(monkeypatch, judge), cm, ["G1", "CHANGED"], samples)
    assert len(judge.inputs) == 1


def test_unparseable_response_not_cached(monkeypatch):
    cm = _FakeCacheManager()
    judge = _GarbageJudge()
    _run(_metric(monkeypatch, judge), cm, ["GEN"], [_sample(5)])
    assert cm.store == {}
    assert len(judge.inputs) == 3          # initial call + 2 retry rounds


def test_reasoning_effort_changes_cache_key(monkeypatch):
    cm = _FakeCacheManager()
    _run(_metric(monkeypatch, _EchoJudge()), cm, ["GEN"], [_sample(5)])
    judge = _EchoJudge()
    _run(
        _metric(monkeypatch, judge, reasoning_effort="low"), cm, ["GEN"], [_sample(5)]
    )
    assert len(judge.inputs) == 1
    assert len(cm.store) == 2
