"""
Generic rubric-driven LLM-as-judge metric for JSON outputs.

A sample contributes one or more "rubric passes" — independent judge calls that
each produce a list of per-item scores. Each pass is self-contained: it carries
its own judge prompt, its own axis definitions, its own pass conditions, and
optionally a metadata list for slicing the results.

The dataset populates `sample.other_args["rubric_passes"]`:

    other_args = {
        "document_type": "...",
        "rubric_passes": [
            {
                "name": "omission",
                "prompt": "<judge system prompt>",
                "id_field": "rubric_id",                      # id key on each judge response item
                "type_field": "type",                         # discriminator — selects axis set
                "axes": {
                    "entity":     [{"field": "semantic_meaning_accuracy", "passes_when": {"eq": 1}}, ...],
                    "structural": [{"field": "schema_correctness",        "passes_when": {"eq": 1}}],
                },
                "metadata": [ {"rubric_id": "R-001", "entity_type": "symptom", ...}, ... ],
                "response_array_key": None,                   # response IS the array
            },
            {
                "name": "commission",
                "prompt": "<judge system prompt>",
                "id_field": "entity_id",
                "type_field": None,
                "axes": {"*": [{"field": "grounding", "passes_when": {"in": ["grounded", "inferred"]}}]},
                "metadata": None,
                "breakdown_fields": ["section", "entity_type", "property"],
                "response_array_key": "entities",
            },
        ],
        "rubric_postprocessor": dataset.postprocess_rubric_results,  # optional callable
    }

Legacy response shape (judge returns `{"rubric_scores": {id: 0/1, ...}}`):

    {
        "name": "rubric_evaluation",
        "prompt": rubric_prompt,
        "response_scores_map_key": "rubric_scores",   # expand to items
        "id_field": "rubric_id",                      # key name on expanded items
        "score_field": "score",                       # value name on expanded items
        "axes": {"*": [{"field": "score", "passes_when": {"eq": 1}}]},
    }

Output shape (flat — pass names at top level):

    {
        "overall_score": 0.xx,          # items passed / items scored across all passes
        "num_questions": N,
        "num_items": K,
        "omission":   {"rubric_score": 0.xx, "num_items": K, "by_axis": {...}, "by_<meta_field>": {...}},
        "commission": {"rubric_score": 0.xx, "num_items": K, "by_axis": {...}, "by_<meta_field>": {...}},
        "document_type_breakdown": {...},   # only when samples span multiple document types
    }

`rubric_score` = fraction of rubric items that passed all their axes. Higher is
better. For the omission pass, "omission: 0.89" means 89% of ground-truth
rubric items were captured correctly, NOT 89% omitted.
"""

import json
import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional

import numpy as np

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.metrics.base_metric_abs import BaseMetric
from karma.models.openai_llm import OpenAILLM
from karma.registries.metrics_registry import register_metric

logger = logging.getLogger(__name__)


_RESERVED_TOP_LEVEL = frozenset({
    "overall_score", "num_questions", "num_items",
    "document_type_breakdown", "std_dev", "bootstrap_std",
    "invalid_yml",
})


@register_metric(
    name="json_rubric_evaluation",
    required_args=["provider_to_use", "model_id"],
    default_args={"provider_to_use": "openai", "model_id": "gpt-4o"},
)
class JsonRubricEvaluationMetric(BaseMetric):
    """Generic rubric-based JSON evaluation via LLM-as-judge."""

    non_percentage_fields = frozenset({"num_questions", "num_items", "count", "invalid_yml"})

    def __init__(
        self,
        metric_name: str,
        provider_to_use: str = "openai",
        model_id: str = "gpt-4o",
        **kwargs,
    ):
        super().__init__(metric_name=metric_name, **kwargs)
        self.provider = provider_to_use
        logger.info(f"Got {provider_to_use} JSON rubric evaluation metric")
        if self.provider == "openai":
            self.model = OpenAILLM(model_name_or_path=model_id, max_tokens=10000)

    # ---------- entrypoint ----------

    def evaluate(self, predictions, references=None, rubrics=None, **kwargs):
        samples = kwargs["samples"]
        serialized_samples = kwargs.get("serialized_samples")
        cache_manager = kwargs.get("cache_manager")
        dataset_name = kwargs.get("dataset_name")
        successes = kwargs.get("successes") or []
        invalid_yml = sum(1 for s in successes if not s)

        logger.info(
            f"Evaluating {len(predictions)} outputs with {self.provider} judge"
        )

        postprocessor = self._resolve_postprocessor(dataset_name)

        # Phase 1: collect every (sample, pass) judge call into a flat batch so
        # OpenAILLM.run's ThreadPoolExecutor can fan them out in parallel.
        jobs: List[Dict[str, Any]] = []
        for sample_idx, (prediction, sample) in enumerate(zip(predictions, samples)):
            other_args = sample.other_args or {}
            rubric_passes = other_args.get("rubric_passes") or []
            if not rubric_passes:
                logger.error(
                    "No 'rubric_passes' on sample.other_args; "
                    "dataset must populate it for json_rubric_evaluation"
                )
            for pass_spec in rubric_passes:
                pass_name = pass_spec["name"]
                if pass_name in _RESERVED_TOP_LEVEL:
                    logger.warning(
                        "Pass name '%s' collides with reserved top-level key; "
                        "it will overwrite or be overwritten during aggregation.",
                        pass_name,
                    )
                jobs.append({
                    "sample_idx": sample_idx,
                    "pass_spec": pass_spec,
                    "eval_input": self._build_judge_input(
                        pass_spec["prompt"], prediction
                    ),
                })

        # Phase 2: single parallel judge dispatch.
        if jobs:
            logger.info(
                "Dispatching %d judge calls for %d samples in parallel",
                len(jobs),
                len(samples),
            )
            judge_responses = self.model.run([j["eval_input"] for j in jobs])
        else:
            judge_responses = []

        # Phase 3: score each response and regroup by sample.
        per_sample_passes: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
        per_sample_metadata: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
        for job, response in zip(jobs, judge_responses):
            sample_idx = job["sample_idx"]
            pass_spec = job["pass_spec"]
            pass_name = pass_spec["name"]
            per_sample_passes[sample_idx][pass_name] = self._score_pass_response(
                response, pass_spec
            )
            per_sample_metadata[sample_idx][pass_name] = pass_spec.get("metadata") or []

        # Phase 4: apply postprocessor and build per-sample records.
        per_sample_records: List[Dict[str, Any]] = []
        for sample_idx, sample in enumerate(samples):
            passes_result = per_sample_passes.get(sample_idx, {})
            metadata_by_pass = per_sample_metadata.get(sample_idx, {})
            if callable(postprocessor):
                try:
                    passes_result = postprocessor(passes_result, metadata_by_pass)
                except Exception as exc:
                    logger.warning("rubric_postprocessor failed: %s", exc)
            document_type = (sample.other_args or {}).get("document_type", "unknown")
            per_sample_records.append({
                "passes": passes_result,
                "document_type": document_type,
            })

        summary = self._aggregate(per_sample_records)
        summary["invalid_yml"] = invalid_yml
        aggregated = {self.metric_name: summary}

        if cache_manager and serialized_samples and dataset_name:
            try:
                self._persist_rubric_results(
                    cache_manager=cache_manager,
                    dataset_name=dataset_name,
                    per_sample_records=per_sample_records,
                    serialized_samples=serialized_samples,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Failed to persist JSON rubric evaluation results: %s", exc
                )

        return aggregated

    @staticmethod
    def _resolve_postprocessor(dataset_name: Optional[str]):
        """Look up the dataset's postprocess_rubric_results via the registry.

        Kept out of `sample.other_args` because callables can't be JSON-
        serialized by the cache layer.
        """
        if not dataset_name:
            return None
        try:
            from karma.registries.dataset_registry import dataset_registry
            cls = dataset_registry.get_dataset_class(dataset_name)
        except Exception:
            return None
        fn = getattr(cls, "postprocess_rubric_results", None)
        return fn if callable(fn) else None

    @staticmethod
    def _build_judge_input(pass_prompt: str, prediction: str) -> DataLoaderIterable:
        complete_prompt = f"{pass_prompt}\n\nGENERATED OUTPUT:\n{prediction}"
        return DataLoaderIterable(
            input=complete_prompt,
            system_prompt="You are an expert evaluator for structured JSON outputs. Respond with strict JSON only.",
        )

    def _score_pass_response(
        self, judge_response: str, pass_spec: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        id_field = pass_spec.get("id_field")
        type_field = pass_spec.get("type_field")
        axes_by_type: Dict[str, List[Dict[str, Any]]] = pass_spec.get("axes") or {}
        metadata_list: List[Dict[str, Any]] = pass_spec.get("metadata") or []
        breakdown_fields: List[str] = pass_spec.get("breakdown_fields") or []
        response_array_key: Optional[str] = pass_spec.get("response_array_key")
        scores_map_key: Optional[str] = pass_spec.get("response_scores_map_key")
        score_field: str = pass_spec.get("score_field", "score")

        parsed = self._parse_json(judge_response)

        if scores_map_key:
            id_key = id_field or "rubric_id"
            scores_map = (
                parsed.get(scores_map_key, {}) if isinstance(parsed, dict) else {}
            )
            if not isinstance(scores_map, dict):
                scores_map = {}
            judge_items = [
                {id_key: k, score_field: v} for k, v in scores_map.items()
            ]
        else:
            judge_items = self._items_from_parsed(parsed, response_array_key)

        metadata_by_id: Dict[str, Dict[str, Any]] = {}
        if id_field and metadata_list:
            for meta in metadata_list:
                mid = meta.get(id_field)
                if mid is not None:
                    metadata_by_id[str(mid)] = meta

        items: List[Dict[str, Any]] = []
        for judge_item in judge_items:
            if not isinstance(judge_item, dict):
                continue

            item_id = judge_item.get(id_field) if id_field else None
            item_type = judge_item.get(type_field) if type_field else None

            # Pick axis set: exact type match, then "*" fallback
            axes = axes_by_type.get(item_type)
            if axes is None:
                axes = axes_by_type.get("*", [])

            axis_scores: Dict[str, int] = {}
            for axis_spec in axes:
                field = axis_spec["field"]
                if field not in judge_item:
                    continue
                passed = self._check_passes(
                    judge_item[field], axis_spec.get("passes_when", {})
                )
                axis_scores[field] = 1 if passed else 0

            item_pass: Optional[bool] = (
                all(v == 1 for v in axis_scores.values())
                if axis_scores
                else None
            )

            # Build metadata for slicing: joined dataset metadata, else
            # selected fields from the judge response itself. When
            # `breakdown_fields` is set on the pass spec, it acts as an
            # allowlist of legitimate aggregation dimensions — fields outside
            # it (raw entity text, cross-references, etc.) are dropped.
            if metadata_by_id and item_id is not None:
                raw_meta = metadata_by_id.get(str(item_id), {})
                if breakdown_fields:
                    item_metadata = {
                        f: raw_meta.get(f)
                        for f in breakdown_fields
                        if f in raw_meta
                    }
                else:
                    item_metadata = dict(raw_meta)
            elif breakdown_fields:
                item_metadata = {
                    f: judge_item.get(f)
                    for f in breakdown_fields
                    if f in judge_item
                }
            else:
                item_metadata = {}

            # id_field is a per-item identifier, not an aggregation dimension.
            if id_field:
                item_metadata.pop(id_field, None)

            items.append({
                "item_id": item_id,
                "type": item_type,
                "axis_scores": axis_scores,
                "item_pass": item_pass,
                "metadata": item_metadata,
                "raw": judge_item,
            })

        return items

    @staticmethod
    def _parse_json(raw_response: str) -> Any:
        clean = (raw_response or "").strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        elif clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse judge response as JSON: %s", e)
            return None

    @staticmethod
    def _items_from_parsed(
        parsed: Any, response_array_key: Optional[str]
    ) -> List[Any]:
        if parsed is None:
            return []
        if response_array_key:
            value = parsed.get(response_array_key) if isinstance(parsed, dict) else None
            return value if isinstance(value, list) else []
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return v
        return []

    @staticmethod
    def _check_passes(value: Any, rule: Dict[str, Any]) -> bool:
        if "eq" in rule:
            return value == rule["eq"]
        if "neq" in rule:
            return value != rule["neq"]
        if "in" in rule:
            return value in rule["in"]
        return False

    # ---------- aggregation ----------

    def _aggregate(
        self, per_sample_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        per_pass_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in per_sample_records:
            for pass_name, items in rec["passes"].items():
                per_pass_items[pass_name].extend(items)

        result: Dict[str, Any] = {}
        total_items = 0
        pass_scores: List[float] = []

        for pass_name, items in per_pass_items.items():
            pass_summary, _passed, n = self._summarise_pass(items)
            total_items += n
            result[pass_name] = pass_summary
            # Top-level shortcut so callers can read `<pass>_score` without
            # reaching into the nested pass summary.
            result[f"{pass_name}_score"] = pass_summary["rubric_score"]
            if n > 0:
                pass_scores.append(pass_summary["rubric_score"])

        # Unweighted mean of per-pass rubric scores so a pass with many items
        # (e.g. commission) doesn't swamp a pass with fewer items (omission).
        result["overall_score"] = (
            float(sum(pass_scores) / len(pass_scores)) if pass_scores else 0.0
        )
        result["num_questions"] = len(per_sample_records)
        result["num_items"] = total_items

        doc_types = {r.get("document_type", "unknown") for r in per_sample_records}
        if len(doc_types) > 1:
            result["document_type_breakdown"] = self._by_document_type(
                per_sample_records
            )

        return result

    @staticmethod
    def _summarise_pass(items: List[Dict[str, Any]]):
        """rubric_score = fraction of rubric items that passed all their axes."""
        scorable = [it for it in items if it["item_pass"] is not None]
        n = len(scorable)
        passed = sum(1 for it in scorable if it["item_pass"])
        rubric_score = float(passed / n) if n > 0 else 0.0

        by_axis: Dict[str, List[int]] = defaultdict(list)
        for it in items:
            for field, score in it["axis_scores"].items():
                by_axis[field].append(int(score))

        axis_summary = {
            field: {
                "pass_rate": float(np.mean(scores)) if scores else 0.0,
                "count": len(scores),
            }
            for field, scores in by_axis.items()
        }

        summary: Dict[str, Any] = {
            "rubric_score": rubric_score,
            "num_items": n,
            "by_axis": axis_summary,
        }

        # Metadata-driven breakdowns: discover fields from each item's
        # attached metadata dict (joined from dataset or pulled from the
        # judge response). Values become group keys.
        # field -> value -> [0/1]
        per_field: Dict[str, Dict[str, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for it in scorable:
            for field, value in (it.get("metadata") or {}).items():
                if value is None:
                    continue
                # Skip list/dict values — not useful for group keys
                if isinstance(value, (list, dict)):
                    continue
                # Skip identifier-like fields (anything ending in _id):
                # per-item identifiers bulk up the output without aggregate
                # insight.
                if field.endswith("_id"):
                    continue
                per_field[field][str(value)].append(
                    1 if it["item_pass"] else 0
                )

        for field, groups in per_field.items():
            # Skip breakdowns where every group has a single item — these are
            # per-item identifiers (e.g. entity text) and add bulk with no
            # aggregate insight.
            if all(len(scores) == 1 for scores in groups.values()):
                continue
            summary[f"by_{field}"] = {
                val: {
                    "pass_rate": float(np.mean(scores)),
                    "count": len(scores),
                }
                for val, scores in sorted(groups.items())
            }

        return summary, passed, n

    @staticmethod
    def _by_document_type(
        per_sample_records: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        buckets: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"pass": 0, "items": 0, "samples": 0}
        )
        for rec in per_sample_records:
            dt = rec.get("document_type", "unknown")
            buckets[dt]["samples"] += 1
            for items in rec["passes"].values():
                for it in items:
                    if it["item_pass"] is None:
                        continue
                    buckets[dt]["items"] += 1
                    if it["item_pass"]:
                        buckets[dt]["pass"] += 1
        return {
            dt: {
                "overall_score": (
                    float(b["pass"] / b["items"]) if b["items"] > 0 else 0.0
                ),
                "num_samples": b["samples"],
                "num_items": b["items"],
            }
            for dt, b in buckets.items()
        }

    # ---------- persistence ----------

    def _persist_rubric_results(
        self,
        *,
        cache_manager,
        dataset_name: str,
        per_sample_records: List[Dict[str, Any]],
        serialized_samples: List[Any],
    ) -> None:
        if not per_sample_records:
            return

        metric_metadata = self._build_metric_metadata()
        records = []
        for index, (rec, serialized_sample) in enumerate(
            zip(per_sample_records, serialized_samples, strict=False)
        ):
            if serialized_sample is None:
                continue

            identifiers = cache_manager.get_cache_identifiers_for_sample(
                serialized_sample
            )

            passes_payload: Dict[str, List[Dict[str, Any]]] = {}
            sample_pass = 0
            sample_items = 0
            for pass_name, items in rec["passes"].items():
                serialised_items = []
                for it in items:
                    serialised_items.append({
                        "item_id": it.get("item_id"),
                        "type": it.get("type"),
                        "axis_scores": it.get("axis_scores", {}),
                        "item_pass": it.get("item_pass"),
                        "metadata": it.get("metadata") or {},
                    })
                    if it.get("item_pass") is not None:
                        sample_items += 1
                        if it["item_pass"]:
                            sample_pass += 1
                passes_payload[pass_name] = serialised_items

            question_score = (
                float(sample_pass / sample_items) if sample_items > 0 else None
            )

            records.append({
                "cache_key": identifiers["cache_key"],
                "dataset_row_hash": identifiers["dataset_row_hash"],
                "dataset_name": dataset_name,
                "config_hash": cache_manager.config_hash,
                "metric_name": self.metric_name,
                "question_index": index,
                "question_score": question_score,
                "rubric_evaluations": {
                    "passes": passes_payload,
                    "document_type": rec.get("document_type"),
                },
                "metric_metadata": metric_metadata,
            })

        if records:
            cache_manager.batch_save_rubric_question_results(records)

    def _build_metric_metadata(self) -> Dict[str, Any]:
        model_identifier = None
        if hasattr(self, "model") and self.model is not None:
            model_identifier = getattr(
                self.model,
                "model_id",
                getattr(self.model, "model_name_or_path", None),
            )

        metadata: Dict[str, Any] = {"provider": self.provider}
        if model_identifier:
            metadata["model_identifier"] = model_identifier
        if hasattr(self, "model") and self.model is not None:
            metadata["model_class"] = self.model.__class__.__name__
        if getattr(self, "metric_args", None):
            metadata["extra_args"] = self.metric_args
        return metadata
