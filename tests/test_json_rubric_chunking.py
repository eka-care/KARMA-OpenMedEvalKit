"""Tests for rubric/entity chunking in json_rubric_evaluation.

Chunking is driven entirely by the metric's ``chunk_size`` (a metric arg — it
never touches the sample/model cache key):

* omission-style pass (has a ``rubrics`` list) — the rubric list is rendered
  into the judge prompt in batches and the per-batch responses are merged.
* commission-style pass (has ``response_array_key`` and no rubrics) — the
  generated note is split into batches of entities so the judge enumerates a
  bounded slice per call, then the entity lists are merged.

These use judge doubles, so no network/model is needed.
"""

import json
import types

from karma.metrics.rubrics.json_rubric_evaluation import JsonRubricEvaluationMetric


_OMISSION_AXES = {
    "entity": [
        {"field": "semantic_meaning_accuracy", "passes_when": {"eq": 1}},
        {"field": "terminology_correctness", "passes_when": {"eq": 1}},
        {"field": "misclassification", "passes_when": {"eq": 0}},
        {"field": "duplication_flag", "passes_when": {"eq": False}},
    ],
    "structural": [{"field": "schema_correctness", "passes_when": {"eq": 1}}],
}
_COMMISSION_AXES = {"*": [{"field": "grounding", "passes_when": {"in": ["grounded", "inferred"]}}]}
_MARKER = "=== RUBRICS TO EVALUATE ==="


def _metric(monkeypatch, judge, chunk_size=10):
    monkeypatch.setattr(
        "karma.metrics.rubrics.json_rubric_evaluation.OpenAILLM", lambda *a, **k: judge
    )
    return JsonRubricEvaluationMetric(
        metric_name="json_rubric_evaluation", provider_to_use="openai",
        model_id="dummy", chunk_size=chunk_size,
    )


# ---------------- omission ----------------

class _EchoJudge:
    """Scores exactly the rubrics injected into each prompt (reads the list it
    is handed). Records every prompt for call-count/injection assertions."""

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
            {"rubric_id": r["rubric_id"], "type": r.get("type", "entity"),
             "semantic_meaning_accuracy": 1, "terminology_correctness": 1,
             "misclassification": 0, "duplication_flag": False, "schema_correctness": 1}
            for r in arr
        ]


class _DroppingJudge(_EchoJudge):
    """Drops the last rubric of every batch, to exercise the coverage check."""

    @staticmethod
    def _score(text):
        scored = _EchoJudge._score(text)
        return scored[:-1] if scored else scored


def _omission_sample(n_rubrics):
    rubrics, metadata = [], []
    for i in range(1, n_rubrics + 1):
        rid = f"R-{i:03d}"
        section = "symptoms" if i <= n_rubrics // 2 else "medicationRequest"
        rubrics.append({"rubric_id": rid, "type": "entity", "evaluation_instruction": f"c{i}"})
        metadata.append({"rubric_id": rid, "section": section})
    pass_spec = {
        "name": "omission", "prompt": "INSTR", "id_field": "rubric_id", "type_field": "type",
        "axes": _OMISSION_AXES, "metadata": metadata, "breakdown_fields": ["section"],
        "response_array_key": None, "rubrics": rubrics,
    }
    return types.SimpleNamespace(
        other_args={"rubric_passes": [pass_spec], "document_type": "note", "language": "en"}
    )


def test_omission_chunks_inject_and_score_every_rubric(monkeypatch):
    judge = _EchoJudge()
    metric = _metric(monkeypatch, judge, chunk_size=10)
    result = metric.evaluate(["GEN"], samples=[_omission_sample(25)], successes=[True])["json_rubric_evaluation"]
    assert len(judge.inputs) == 3                       # 25 / 10 -> 3 batches
    assert all(_MARKER in p for p in judge.inputs)      # rubrics injected each call
    assert result["omission"]["num_items"] == 25        # merged
    assert result["omission"]["by_section"]["medicationRequest"]["count"] == 13


def test_omission_single_batch_when_chunk_size_exceeds_list(monkeypatch):
    judge = _EchoJudge()
    metric = _metric(monkeypatch, judge, chunk_size=100)
    result = metric.evaluate(["GEN"], samples=[_omission_sample(25)], successes=[True])["json_rubric_evaluation"]
    assert len(judge.inputs) == 1
    assert result["omission"]["num_items"] == 25


def test_no_injection_when_chunk_size_zero(monkeypatch):
    """chunk_size=0 disables chunking/injection entirely (single call, no rubric
    block) — the judge double then scores nothing, since it only sees rubrics
    it is given."""
    judge = _EchoJudge()
    metric = _metric(monkeypatch, judge, chunk_size=0)
    result = metric.evaluate(["GEN"], samples=[_omission_sample(25)], successes=[True])["json_rubric_evaluation"]
    assert len(judge.inputs) == 1
    assert _MARKER not in judge.inputs[0]
    assert result["omission"]["num_items"] == 0


def test_coverage_is_one_when_all_rubrics_scored(monkeypatch):
    metric = _metric(monkeypatch, _EchoJudge(), chunk_size=10)
    result = metric.evaluate(["GEN"], samples=[_omission_sample(25)], successes=[True])["json_rubric_evaluation"]
    assert result["omission_coverage"] == 1.0


def test_coverage_flags_dropped_rubrics(monkeypatch, caplog):
    metric = _metric(monkeypatch, _DroppingJudge(), chunk_size=10)
    with caplog.at_level("WARNING"):
        result = metric.evaluate(["GEN"], samples=[_omission_sample(25)], successes=[True])["json_rubric_evaluation"]
    assert result["omission"]["num_items"] == 22        # 25 - 3 dropped
    assert result["omission_coverage"] == 22 / 25
    assert any("missing" in rec.message for rec in caplog.records)


def test_support_count_is_presented_not_scored(monkeypatch):
    """Support (count) per field = rubrics presented for that field, even when
    the judge drops some — not the number it scored."""
    metric = _metric(monkeypatch, _DroppingJudge(), chunk_size=10)
    result = metric.evaluate(["GEN"], samples=[_omission_sample(25)], successes=[True])["json_rubric_evaluation"]
    by_section = result["omission"]["by_section"]
    # 25 rubrics: symptoms=12, medicationRequest=13 — presented, regardless of drops
    assert by_section["symptoms"]["count"] == 12
    assert by_section["medicationRequest"]["count"] == 13
    # but the judge dropped 3, so fewer were actually scored (coverage < 1)
    assert result["omission"]["num_items"] == 22
    assert result["omission_coverage"] == 22 / 25


# ---------------- commission ----------------

_GROUND_MARKER = "=== ENTITIES TO GROUND ==="


class _GroundJudge:
    """Grounds every listed entity_id — returns compact {entity_id, grounding},
    reading only the code-supplied rubric list (never enumerates itself)."""

    def __init__(self):
        self.inputs = []

    def run(self, inputs):
        out = []
        for item in inputs:
            self.inputs.append(item.input)
            block = item.input.split(_GROUND_MARKER, 1)[1]
            arr = json.loads(block[block.find("["): block.rfind("]") + 1])
            out.append(json.dumps({"entities": [
                {"entity_id": e["entity_id"], "grounding": "grounded"} for e in arr
            ]}))
        return out


def _stub_build_commission_scoring(pass_spec, prediction):
    """Test double for the dataset hook: enumerate list items of a JSON note as
    entity rubrics + a grounding header."""
    note = json.loads(prediction)
    rubrics = []
    for section, val in note.items():
        if isinstance(val, list):
            for it in val:
                rubrics.append({
                    "entity_id": f"C-{len(rubrics) + 1:04d}", "section": section,
                    "entity_type": section, "property": "main",
                    "description": f"{section}: {it}",
                })
    return "GROUND THESE\n=== ORIGINAL TRANSCRIPT ===\nT", rubrics


def _commission_sample(n_entities):
    note = {"symptoms": [{"name": f"s{i}"} for i in range(n_entities)]}
    pass_spec = {
        "name": "commission", "prompt": "COMM\n=== ORIGINAL TRANSCRIPT ===\nT",
        "id_field": "entity_id", "type_field": None, "axes": _COMMISSION_AXES,
        "metadata": None, "breakdown_fields": ["section"], "response_array_key": "entities",
    }
    sample = types.SimpleNamespace(
        other_args={"rubric_passes": [pass_spec], "document_type": "note", "language": "en"}
    )
    return sample, json.dumps(note)


def _commission_metric(monkeypatch, judge, chunk_size=10):
    metric = _metric(monkeypatch, judge, chunk_size=chunk_size)
    monkeypatch.setattr(
        metric, "_resolve_dataset_callable",
        lambda dn, attr: _stub_build_commission_scoring if attr == "build_commission_scoring" else None,
    )
    return metric


def test_commission_code_enumerates_chunks_and_scores(monkeypatch):
    judge = _GroundJudge()
    metric = _commission_metric(monkeypatch, judge, chunk_size=10)
    sample, pred = _commission_sample(25)
    result = metric.evaluate([pred], samples=[sample], successes=[True],
                             dataset_name="x")["json_rubric_evaluation"]
    assert len(judge.inputs) == 3                            # 25 / 10 -> 3 batches
    assert all(_GROUND_MARKER in p for p in judge.inputs)    # entity list injected
    assert all("GENERATED OUTPUT:" not in p for p in judge.inputs)  # no note appended
    assert result["commission"]["num_items"] == 25          # every entity scored, merged
    assert result["commission_coverage"] == 1.0             # coverage now works for commission


def test_commission_single_call_when_small(monkeypatch):
    judge = _GroundJudge()
    metric = _commission_metric(monkeypatch, judge, chunk_size=100)
    sample, pred = _commission_sample(25)
    result = metric.evaluate([pred], samples=[sample], successes=[True],
                             dataset_name="x")["json_rubric_evaluation"]
    assert len(judge.inputs) == 1
    assert result["commission"]["num_items"] == 25
