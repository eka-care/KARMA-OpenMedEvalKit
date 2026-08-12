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
        "overall_score": 0.xx,          # 1 - mean(omission_rate, commission_rate)
        "num_questions": N,
        "num_items": K,
        "omission_rate":   0.xx,        # = 1 - omission rubric_score (fraction missed)
        "commission_rate": 0.xx,        # = 1 - commission rubric_score (fraction hallucinated)
        "semantic_meaning_accuracy": 0.xx,   # axis pass rate from the omission pass
        "terminology_correctness":   0.xx,   # axis pass rate from the omission pass
        "omission":   {"rubric_score": 0.xx, "num_items": K, "by_axis": {...}, "by_<meta_field>": {...}},
        "commission": {"rubric_score": 0.xx, "num_items": K, "by_axis": {...}, "by_<meta_field>": {...}},
        "by_language":              {...},  # only when samples carry a language label
        "document_type_breakdown":  {...},  # only when samples span multiple document types
    }

`rubric_score` = fraction of rubric items that passed all their axes. Higher is
better. ``omission_rate`` and ``commission_rate`` are reported as failure
rates (1 - rubric_score) so larger numbers mean *more* omitted /
hallucinated content respectively, while ``overall_score`` keeps the
"higher is better" convention.
"""

import json
import logging
import re
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
    "document_type_breakdown", "by_language",
    "omission_rate", "commission_rate",
    "omission_coverage", "commission_coverage",
    "semantic_meaning_accuracy", "terminology_correctness",
    "std_dev", "bootstrap_std",
    "invalid_yml",
})

# Axes from the omission pass that we surface at the top level alongside
# omission_rate / commission_rate. They stay as positive pass rates
# (1.0 == fully correct) since "accuracy" / "correctness" naturally read
# that way; only the omission/commission scores are inverted into rates.
_TOP_LEVEL_OMISSION_AXES = ("semantic_meaning_accuracy", "terminology_correctness")


@register_metric(
    name="json_rubric_evaluation",
    required_args=["provider_to_use", "model_id"],
    default_args={"provider_to_use": "openai", "model_id": "gpt-4o", "chunk_size": 50},
)
class JsonRubricEvaluationMetric(BaseMetric):
    """Generic rubric-based JSON evaluation via LLM-as-judge."""

    non_percentage_fields = frozenset({"num_questions", "num_items", "count", "invalid_yml"})

    def __init__(
        self,
        metric_name: str,
        provider_to_use: str = "openai",
        model_id: str = "gpt-4o",
        chunk_size: int = 50,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(metric_name=metric_name, **kwargs)
        self.provider = provider_to_use
        # Rubric chunking: when a pass sets ``chunk_rubrics``, its rubric list is
        # rendered into the judge prompt in batches of this size (separate judge
        # calls, merged afterwards) so long lists are scored reliably. <= 0 means
        # a single batch.
        self.chunk_size = int(chunk_size) if chunk_size else 0
        self.reasoning_effort = reasoning_effort
        logger.info(f"Got {provider_to_use} JSON rubric evaluation metric")
        if self.provider == "openai":
            judge_kwargs = {"model_name_or_path": model_id, "max_tokens": 32768}
            # Judge verdicts are simple pass/fail calls; lowering the judge's
            # reasoning effort (e.g. "low") cuts its output-token cost several-
            # fold. Validated against a full-effort baseline before use.
            if reasoning_effort:
                judge_kwargs["reasoning_effort"] = reasoning_effort
            self.model = OpenAILLM(**judge_kwargs)

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
        commission_builder = self._resolve_dataset_callable(
            dataset_name, "build_commission_scoring"
        )

        # Phase 1: collect every (sample, pass) judge call into a flat batch so
        # OpenAILLM.run's ThreadPoolExecutor can fan them out in parallel.
        jobs: List[Dict[str, Any]] = []
        # (sample_idx, pass_name) -> the rubric list actually sent, for coverage.
        effective_rubrics: Dict[tuple, List[Dict[str, Any]]] = {}
        # pass_name -> presented "support" counts (# rubrics per field/axis), so
        # the reported count is the field's size, not how many the judge scored.
        presented_counts: Dict[str, Dict[str, Any]] = {}
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
                # Commission: when the dataset provides build_commission_scoring,
                # code enumerates the entities/properties to ground and the judge
                # only classifies each — chunked in batches like omission.
                if commission_builder and pass_name == "commission":
                    header, com_rubrics = commission_builder(pass_spec, prediction)
                    id_field = pass_spec.get("id_field") or "entity_id"
                    effective_rubrics[(sample_idx, pass_name)] = com_rubrics
                    self._accumulate_presented(presented_counts, pass_name, com_rubrics, pass_spec)
                    for chunk_idx, chunk_prompt in self._commission_prompts(
                        header, com_rubrics, id_field
                    ):
                        jobs.append({
                            "sample_idx": sample_idx, "pass_spec": pass_spec,
                            "chunk_idx": chunk_idx,
                            "eval_input": self._build_judge_input(chunk_prompt),
                        })
                    continue
                # Omission (and any other pass): chunk a static rubric list, or a
                # single call. chunk_size drives batching, so the cache key is
                # untouched and other datasets behave exactly as before.
                if pass_spec.get("rubrics"):
                    effective_rubrics[(sample_idx, pass_name)] = pass_spec["rubrics"]
                # Support = presented rubrics per field; omission carries these on
                # its metadata (section/type/property per rubric_id).
                self._accumulate_presented(
                    presented_counts, pass_name, pass_spec.get("metadata") or [], pass_spec
                )
                for chunk_idx, chunk_prompt, chunk_pred in self._pass_prompts(
                    pass_spec, prediction
                ):
                    jobs.append({
                        "sample_idx": sample_idx, "pass_spec": pass_spec,
                        "chunk_idx": chunk_idx,
                        "eval_input": self._build_judge_input(chunk_prompt, chunk_pred),
                    })

        judge_hashes: List[Optional[str]] = [None] * len(jobs)
        cached_by_hash: Dict[str, Dict[str, Any]] = {}
        if cache_manager and jobs:
            jm_config = self._build_judge_model_config()
            for i, job in enumerate(jobs):
                judge_hashes[i] = cache_manager.generate_hash({
                    "eval_input": job["eval_input"].model_dump(),
                    "judge_model_config": jm_config,
                })
            try:
                cached_by_hash = cache_manager.get_rubric_outputs(
                    [h for h in judge_hashes if h]
                ) or {}
            except Exception as exc:
                logger.warning("Rubric cache lookup failed: %s", exc)
                cached_by_hash = {}

        # Phase 2: parallel judge dispatch, cache hits seeded first.
        if jobs:
            judge_responses = [None] * len(jobs)
            for i, job in enumerate(jobs):
                cached = cached_by_hash.get(judge_hashes[i]) if judge_hashes[i] else None
                raw = (cached or {}).get("raw_response")
                if raw and self._parse_json(raw) is not None:
                    judge_responses[i] = raw
                    job["from_cache"] = True
            miss_indices = [i for i, r in enumerate(judge_responses) if r is None]
            if cached_by_hash:
                logger.info(
                    "[%s] judge cache: %d/%d reused",
                    self.metric_name, len(jobs) - len(miss_indices), len(jobs),
                )
            if miss_indices:
                logger.info(
                    "Dispatching %d judge calls for %d samples in parallel",
                    len(miss_indices),
                    len(samples),
                )
                fresh = self.model.run([jobs[i]["eval_input"] for i in miss_indices])
                for i, resp in zip(miss_indices, fresh):
                    judge_responses[i] = resp
            # A truncated array (judge ran out of completion budget) or a
            # transient API error loses a whole chunk of rubrics. Retry just
            # the unparseable responses before scoring so coverage holds.
            for attempt in range(2):
                bad = [
                    i for i, resp in enumerate(judge_responses)
                    if self._parse_json(resp) is None
                ]
                if not bad:
                    break
                logger.warning(
                    "Retrying %d unparseable judge responses (attempt %d)",
                    len(bad), attempt + 1,
                )
                retried = self.model.run([jobs[i]["eval_input"] for i in bad])
                for i, resp in zip(bad, retried):
                    if self._parse_json(resp) is not None:
                        judge_responses[i] = resp
        else:
            judge_responses = []

        # Phase 3: regroup responses by (sample, pass) — a chunked pass emits
        # several responses that must be merged before scoring — then score
        # each pass once over its combined judge output.
        per_sample_passes: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
        per_sample_metadata: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
        grouped: Dict[tuple, List[tuple]] = defaultdict(list)
        pass_spec_by_key: Dict[tuple, Dict[str, Any]] = {}
        for job, response in zip(jobs, judge_responses):
            key = (job["sample_idx"], job["pass_spec"]["name"])
            grouped[key].append((job.get("chunk_idx", 0), response))
            pass_spec_by_key[key] = job["pass_spec"]
        # [returned, presented] rubric counts per pass — completeness check for
        # chunk_rubrics passes, surfaced as <pass>_coverage below.
        coverage_tally: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        for (sample_idx, pass_name), pairs in grouped.items():
            responses = [resp for _, resp in sorted(pairs, key=lambda p: p[0])]
            pass_spec = pass_spec_by_key[(sample_idx, pass_name)]
            items = self._score_pass_response(responses, pass_spec)
            per_sample_passes[sample_idx][pass_name] = items
            per_sample_metadata[sample_idx][pass_name] = pass_spec.get("metadata") or []
            eff = effective_rubrics.get((sample_idx, pass_name))
            if eff is not None:
                self._track_rubric_coverage(
                    coverage_tally, pass_name, eff,
                    pass_spec.get("id_field") or "rubric_id", items, sample_idx,
                )

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
            sample_other = sample.other_args or {}
            document_type = sample_other.get("document_type", "unknown")
            language = sample_other.get("language") or "unknown"
            per_sample_records.append({
                "passes": passes_result,
                "document_type": document_type,
                "language": language,
            })

        summary = self._aggregate(per_sample_records, presented_counts)
        summary["invalid_yml"] = invalid_yml
        # Completeness: fraction of presented rubrics the judge actually scored,
        # per chunk_rubrics pass. 1.0 == every rubric came back; < 1.0 means the
        # judge dropped some (also logged per-sample above).
        for pass_name, (returned, presented) in coverage_tally.items():
            if presented > 0:
                summary[f"{pass_name}_coverage"] = float(returned / presented)
                if returned < presented:
                    logger.warning(
                        "%s: judge scored %d/%d rubrics across %d samples "
                        "(coverage %.3f) — some rubrics were dropped",
                        pass_name, returned, presented,
                        len(per_sample_records), returned / presented,
                    )
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

        if cache_manager and serialized_samples and jobs:
            try:
                self._persist_rubric_outputs(
                    cache_manager=cache_manager,
                    jobs=jobs,
                    judge_hashes=judge_hashes,
                    judge_responses=judge_responses,
                    per_sample_passes=per_sample_passes,
                    serialized_samples=serialized_samples,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to persist judge outputs: %s", exc)

        return aggregated

    @staticmethod
    def _resolve_dataset_callable(dataset_name: Optional[str], attr: str):
        """Look up a callable ``attr`` on the dataset class via the registry.

        Used for dataset hooks (postprocess_rubric_results,
        build_commission_scoring) that can't live in ``sample.other_args``
        because callables aren't JSON-serializable for the cache layer.
        """
        if not dataset_name:
            return None
        try:
            from karma.registries.dataset_registry import dataset_registry
            cls = dataset_registry.get_dataset_class(dataset_name)
        except Exception:
            return None
        fn = getattr(cls, attr, None)
        return fn if callable(fn) else None

    @classmethod
    def _resolve_postprocessor(cls, dataset_name: Optional[str]):
        return cls._resolve_dataset_callable(dataset_name, "postprocess_rubric_results")

    @staticmethod
    def _build_judge_input(pass_prompt: str, prediction: Optional[str] = None) -> DataLoaderIterable:
        # The generated output is appended for passes that score it (omission).
        # Code-enumerated commission passes it as None — the items to check are
        # already rendered into the prompt.
        complete_prompt = (
            pass_prompt if prediction is None
            else f"{pass_prompt}\n\nGENERATED OUTPUT:\n{prediction}"
        )
        return DataLoaderIterable(
            input=complete_prompt,
            system_prompt="You are an expert evaluator for structured JSON outputs. Respond with strict JSON only.",
        )

    def _pass_prompts(self, pass_spec: Dict[str, Any], prediction: str):
        """Yield ``(chunk_idx, prompt, prediction)`` judge inputs for a pass with
        a static ``rubrics`` list (omission).

        Batching is driven only by ``self.chunk_size`` — a metric arg, so it
        never touches the sample/model cache key. The rubric list is rendered
        into the prompt in batches; the prediction is sent whole each call.
        Falls back to a single ``(0, prompt, prediction)`` when there is nothing
        to chunk, so other passes/datasets behave exactly as before.
        """
        base_prompt = pass_spec["prompt"]
        size = self.chunk_size
        id_field = pass_spec.get("id_field") or "rubric_id"
        rubrics = pass_spec.get("rubrics") or []
        if size and size > 0 and rubrics:
            for i, start in enumerate(range(0, len(rubrics), size)):
                batch = rubrics[start:start + size]
                yield i, base_prompt + self._render_items(batch, id_field, "RUBRICS TO EVALUATE"), prediction
            return
        yield 0, base_prompt, prediction

    def _commission_prompts(self, header: str, rubrics: List[Dict[str, Any]], id_field: str):
        """Yield ``(chunk_idx, prompt)`` for a code-enumerated commission pass:
        the entity/property rubric list rendered into the grounding prompt in
        batches of ``self.chunk_size``. No generated output is appended — the
        items to check ARE the rubrics, grounded against the transcript already
        in ``header``.
        """
        size = self.chunk_size
        if not (size and size > 0) or len(rubrics) <= size:
            yield 0, header + self._render_items(rubrics, id_field, "ENTITIES TO GROUND")
            return
        for i, start in enumerate(range(0, len(rubrics), size)):
            batch = rubrics[start:start + size]
            yield i, header + self._render_items(batch, id_field, "ENTITIES TO GROUND")

    @staticmethod
    def _render_items(items: List[Dict[str, Any]], id_field: str, label: str) -> str:
        """Render a batch of rubric/entity items as a labelled JSON block. The
        pass/scoring prompt explains how to read each; this appends the actual
        list, keyed by ``id_field``, for the judge to score."""
        return (
            f"\n\n=== {label} ===\n"
            f"Evaluate EVERY item in this list. Return exactly one result object "
            f"per {id_field}, in the same order. Do not skip any, and do not invent "
            f"{id_field}s that are not listed.\n"
            + json.dumps(items, ensure_ascii=False, indent=2)
        )

    @staticmethod
    def _track_rubric_coverage(tally, pass_name, rubrics, id_field, items, sample_idx):
        """Check that every presented rubric/entity came back scored. Updates
        ``tally[pass_name]`` = ``[returned, presented]`` and warns for any sample
        whose judge output dropped items — so a shortfall is never silent."""
        if not rubrics:
            return
        presented = {
            str(r.get(id_field)) for r in rubrics if r.get(id_field) is not None
        }
        if not presented:
            return
        scored = {
            str(it["item_id"]) for it in items if it.get("item_pass") is not None
        }
        covered = presented & scored
        tally[pass_name][0] += len(covered)
        tally[pass_name][1] += len(presented)
        missing = presented - scored
        if missing:
            logger.warning(
                "%s: judge scored %d/%d items for sample %s; %d missing (e.g. %s)",
                pass_name, len(covered), len(presented), sample_idx, len(missing),
                ", ".join(sorted(missing)[:5]),
            )

    def _score_pass_response(
        self, judge_responses, pass_spec: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        id_field = pass_spec.get("id_field")
        type_field = pass_spec.get("type_field")
        axes_by_type: Dict[str, List[Dict[str, Any]]] = pass_spec.get("axes") or {}
        metadata_list: List[Dict[str, Any]] = pass_spec.get("metadata") or []
        breakdown_fields: List[str] = pass_spec.get("breakdown_fields") or []
        response_array_key: Optional[str] = pass_spec.get("response_array_key")
        scores_map_key: Optional[str] = pass_spec.get("response_scores_map_key")
        score_field: str = pass_spec.get("score_field", "score")

        # A pass may arrive as several chunk responses (chunk_rubrics). Accept a
        # single string for backward compatibility and merge the judge items
        # parsed from every response before scoring.
        if isinstance(judge_responses, str):
            judge_responses = [judge_responses]

        judge_items: List[Any] = []
        for judge_response in judge_responses:
            parsed = self._parse_json(judge_response)
            if scores_map_key:
                id_key = id_field or "rubric_id"
                scores_map = (
                    parsed.get(scores_map_key, {}) if isinstance(parsed, dict) else {}
                )
                if not isinstance(scores_map, dict):
                    scores_map = {}
                judge_items.extend(
                    {id_key: k, score_field: v} for k, v in scores_map.items()
                )
            else:
                judge_items.extend(
                    self._items_from_parsed(parsed, response_array_key)
                )

        # Compact judge output: verdicts may arrive as positional arrays instead
        # of objects, to save output tokens. Two shapes are supported:
        #   * a pass with an ``array_schema`` (multi-axis, type-discriminated) —
        #     e.g. ["R-001","e",1,1,0,0] / ["S-001","s",1] expands to the named
        #     axis object keyed by id_field/type_field.
        #   * a plain 2-element ``[id, bool]`` pair -> {id_field: id, "pass": bool}.
        # Object responses pass through untouched, so other datasets/formats are
        # unaffected.
        array_schema = pass_spec.get("array_schema")
        id_key = id_field or "id"
        judge_items = [
            self._expand_array_item(it, array_schema, id_key)
            if isinstance(it, (list, tuple)) else it
            for it in judge_items
        ]

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
        raw = (raw_response or "").strip()
        if not raw:
            return None
        # Judges (Claude in particular) may wrap the JSON in a prose preamble
        # and/or a markdown code fence, e.g.
        #   "I'll evaluate each rubric:\n\n```json\n[ ... ]\n```".
        # Prefer the contents of a fenced block; otherwise use the raw text.
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
        candidate = (fence.group(1) if fence else raw).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # Last resort: slice the outermost JSON array/object out of the text,
        # tolerating leading/trailing prose the fence regex didn't catch.
        for opener, closer in (("[", "]"), ("{", "}")):
            i, j = candidate.find(opener), candidate.rfind(closer)
            if 0 <= i < j:
                try:
                    return json.loads(candidate[i:j + 1])
                except json.JSONDecodeError:
                    continue
        logger.error("Failed to parse judge response as JSON: %r", candidate[:120])
        return None

    @staticmethod
    def _expand_array_item(item, array_schema, id_key):
        """Expand a positional-array verdict into the object form the scorer
        expects. With an ``array_schema`` (multi-axis, type-discriminated):

            array_schema = {
              "id_index": 0, "type_index": 1,
              "id_field": "rubric_id", "type_field": "type",
              "by_type": {"e": {"type": "entity",
                                "fields": ["semantic_meaning_accuracy", ...]},
                          "s": {"type": "structural",
                                "fields": ["schema_correctness"]}},
            }

        Without a schema, a plain ``[id, verdict]`` pair becomes
        ``{id_key: id, "pass": bool(verdict)}``. Anything that doesn't fit the
        expected shape is returned unchanged (skipped later by the scorer).
        """
        if array_schema and len(item) >= 2:
            id_i = array_schema.get("id_index", 0)
            ty_i = array_schema.get("type_index", 1)
            code = item[ty_i]
            spec = (array_schema.get("by_type") or {}).get(code)
            if spec is not None:
                obj = {
                    array_schema.get("id_field", id_key): item[id_i],
                    array_schema.get("type_field", "type"): spec.get("type", code),
                }
                values = [v for i, v in enumerate(item) if i not in (id_i, ty_i)]
                for field, val in zip(spec.get("fields", []), values):
                    obj[field] = val
                return obj
            return item
        if len(item) == 2:
            return {id_key: item[0], "pass": bool(item[1])}
        return item

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
        self, per_sample_records: List[Dict[str, Any]],
        presented_counts: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        presented_counts = presented_counts or {}
        per_pass_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in per_sample_records:
            for pass_name, items in rec["passes"].items():
                per_pass_items[pass_name].extend(items)

        result: Dict[str, Any] = {}
        total_items = 0
        pass_rates: List[float] = []

        for pass_name, items in per_pass_items.items():
            pass_summary, _passed, n = self._summarise_pass(
                items, presented_counts.get(pass_name)
            )
            total_items += n
            result[pass_name] = pass_summary
            # Surface the failure rate (1 - rubric_score) at the top level so
            # ``omission_rate`` / ``commission_rate`` read as "fraction
            # missed / hallucinated".
            rate = 1.0 - pass_summary["rubric_score"]
            result[f"{pass_name}_rate"] = rate
            if n > 0:
                pass_rates.append(rate)

        # Promote the omission pass's accuracy axes to top-level so they sit
        # next to ``omission_rate`` / ``commission_rate``. Kept as positive
        # pass rates since "accuracy"/"correctness" read that way naturally.
        omission_summary = result.get("omission") or {}
        for axis_name in _TOP_LEVEL_OMISSION_AXES:
            axis_summary = (omission_summary.get("by_axis") or {}).get(axis_name)
            if axis_summary:
                result[axis_name] = axis_summary["pass_rate"]

        # overall_score keeps the "higher is better" convention:
        # 1 - mean(per-pass failure rates) == mean(per-pass rubric_scores).
        # Unweighted across passes so commission (many items) doesn't swamp
        # omission (fewer items).
        result["overall_score"] = (
            float(1.0 - sum(pass_rates) / len(pass_rates)) if pass_rates else 0.0
        )
        result["num_questions"] = len(per_sample_records)
        result["num_items"] = total_items

        languages = {r.get("language", "unknown") for r in per_sample_records}
        # Skip the breakdown when every sample is the unlabelled bucket — it
        # would just duplicate the top-level numbers. Show it whenever any
        # row carries a real language tag, even if there's only one.
        if languages and languages != {"unknown"}:
            by_language = self._by_language(per_sample_records)
            result["by_language"] = by_language
            # Promote each language's overall_score to a flat ``<lang>_score``
            # key so callers can read it next to ``omission_rate`` /
            # ``commission_rate`` without descending into ``by_language``.
            for lang, entry in by_language.items():
                if lang == "unknown":
                    continue
                result[f"{lang}_score"] = entry["overall_score"]

        doc_types = {r.get("document_type", "unknown") for r in per_sample_records}
        if len(doc_types) > 1:
            result["document_type_breakdown"] = self._by_document_type(
                per_sample_records
            )

        return result

    @staticmethod
    def _accumulate_presented(presented_counts, pass_name, rubrics, pass_spec):
        """Tally how many rubrics were *presented* per field / axis (the field's
        support / size), so the reported ``count`` is the number of rubrics for
        that field, not the number the judge happened to score."""
        if not rubrics:
            return
        axes_by_type = pass_spec.get("axes") or {}
        type_field = pass_spec.get("type_field")
        breakdown_fields = pass_spec.get("breakdown_fields") or []
        pc = presented_counts.setdefault(
            pass_name,
            {"by_field": defaultdict(lambda: defaultdict(int)), "by_axis": defaultdict(int)},
        )
        for r in rubrics:
            for field in breakdown_fields:
                value = r.get(field)
                if value is None or isinstance(value, (list, dict)) or field.endswith("_id"):
                    continue
                pc["by_field"][field][str(value)] += 1
            rtype = r.get(type_field) if type_field else None
            axes = axes_by_type.get(rtype)
            if axes is None:
                axes = axes_by_type.get("*", [])
            for axis in axes:
                pc["by_axis"][axis["field"]] += 1

    @staticmethod
    def _summarise_pass(items: List[Dict[str, Any]], presented=None):
        """rubric_score = fraction of scored rubric items that passed all axes.

        ``count`` (support) in each breakdown is the number of rubrics
        *presented* for that field/axis when known (from ``presented``); the
        ``pass_rate`` is still over what the judge scored. Falls back to the
        scored count when no presented tally is supplied.
        """
        scorable = [it for it in items if it["item_pass"] is not None]
        n = len(scorable)
        passed = sum(1 for it in scorable if it["item_pass"])
        rubric_score = float(passed / n) if n > 0 else 0.0

        pres_axis = (presented or {}).get("by_axis") or {}
        pres_field = (presented or {}).get("by_field") or {}

        # scored tallies drive pass_rate
        by_axis: Dict[str, List[int]] = defaultdict(list)
        for it in items:
            for field, score in it["axis_scores"].items():
                by_axis[field].append(int(score))

        axis_summary = {
            field: {
                "pass_rate": float(np.mean(by_axis[field])) if by_axis.get(field) else 0.0,
                "count": pres_axis.get(field, len(by_axis.get(field, []))),
            }
            for field in sorted(set(by_axis) | set(pres_axis))
        }

        summary: Dict[str, Any] = {
            "rubric_score": rubric_score,
            "num_items": n,
            "by_axis": axis_summary,
        }

        # Metadata-driven breakdowns (section / type / property): pass_rate over
        # scored items, count = rubrics presented for that value.
        per_field: Dict[str, Dict[str, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for it in scorable:
            for field, value in (it.get("metadata") or {}).items():
                if value is None or isinstance(value, (list, dict)) or field.endswith("_id"):
                    continue
                per_field[field][str(value)].append(1 if it["item_pass"] else 0)

        for field in sorted(set(per_field) | set(pres_field)):
            scored_groups = per_field.get(field, {})
            pres_groups = pres_field.get(field) or {}
            # Skip per-item-id-like breakdowns where every group has one rubric.
            gauge = pres_groups if pres_groups else {v: [1] for v in scored_groups}
            if gauge and all((c if pres_groups else len(c)) == 1 for c in gauge.values()):
                continue
            summary[f"by_{field}"] = {
                val: {
                    "pass_rate": float(np.mean(scored_groups[val])) if scored_groups.get(val) else 0.0,
                    "count": pres_groups.get(val, len(scored_groups.get(val, []))),
                }
                for val in sorted(set(scored_groups) | set(pres_groups))
            }

        return summary, passed, n

    @staticmethod
    def _by_language(
        per_sample_records: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Per-language rollup with the same rate semantics as the top level.

        For each language bucket we recompute rubric_score per pass (passed /
        scorable items in that bucket), invert into ``<pass>_rate``, and
        derive ``overall_score = 1 - mean(per-pass rates)`` so the per-language
        numbers line up with the top-level summary.
        """
        # language -> pass_name -> {"pass": int, "items": int}
        buckets: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"pass": 0, "items": 0})
        )
        # language -> sample count, total scorable items
        sample_counts: Dict[str, int] = defaultdict(int)
        item_counts: Dict[str, int] = defaultdict(int)
        for rec in per_sample_records:
            lang = rec.get("language", "unknown")
            sample_counts[lang] += 1
            for pass_name, items in rec["passes"].items():
                bucket = buckets[lang][pass_name]
                for it in items:
                    if it["item_pass"] is None:
                        continue
                    bucket["items"] += 1
                    item_counts[lang] += 1
                    if it["item_pass"]:
                        bucket["pass"] += 1

        out: Dict[str, Dict[str, Any]] = {}
        for lang, passes in buckets.items():
            entry: Dict[str, Any] = {
                "num_samples": sample_counts[lang],
                "num_items": item_counts[lang],
            }
            rates: List[float] = []
            for pass_name, b in passes.items():
                if b["items"] == 0:
                    continue
                pass_rubric_score = b["pass"] / b["items"]
                rate = 1.0 - pass_rubric_score
                entry[f"{pass_name}_rate"] = float(rate)
                rates.append(rate)
            entry["overall_score"] = (
                float(1.0 - sum(rates) / len(rates)) if rates else 0.0
            )
            out[lang] = entry
        return dict(sorted(out.items()))

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

    def _build_judge_model_config(self) -> Dict[str, Any]:
        """Whitelisted judge identity for the rubric_outputs cache key."""
        model = getattr(self, "model", None)
        model_id = getattr(model, "model_id", None) or getattr(
            model, "model_name_or_path", None
        )
        config: Dict[str, Any] = {"provider": self.provider, "model_id": model_id}
        if self.reasoning_effort:
            config["reasoning_effort"] = self.reasoning_effort
        return config

    def _persist_rubric_outputs(
        self,
        *,
        cache_manager,
        jobs: List[Dict[str, Any]],
        judge_hashes: List[Optional[str]],
        judge_responses: List[Any],
        per_sample_passes: Dict[int, Dict[str, List[Dict[str, Any]]]],
        serialized_samples: List[Any],
    ) -> None:
        """Write one content-addressed rubric_outputs row per fresh judge call."""
        jm_config = self._build_judge_model_config()
        records = []
        for job, judge_hash, response in zip(jobs, judge_hashes, judge_responses):
            if not judge_hash or job.get("from_cache"):
                continue
            if not response or self._parse_json(response) is None:
                continue
            sample_idx = job["sample_idx"]
            serialized = (
                serialized_samples[sample_idx]
                if sample_idx < len(serialized_samples)
                else None
            )
            if serialized is None:
                continue
            identifiers = cache_manager.get_cache_identifiers_for_sample(serialized)
            pass_name = job["pass_spec"]["name"]
            chunk_idx = job.get("chunk_idx", 0)
            criteria = []
            if chunk_idx == 0:
                # merged pass items go on the first chunk's row only
                items = (per_sample_passes.get(sample_idx) or {}).get(pass_name) or []
                criteria = [
                    {
                        "criteria_met": bool(it.get("item_pass")),
                        "explanation": json.dumps(
                            {
                                "axis_scores": it.get("axis_scores", {}),
                                "type": it.get("type"),
                            },
                            ensure_ascii=False,
                        ),
                        "rubric": {"criterion": str(it.get("item_id"))},
                    }
                    for it in items
                    if it.get("item_pass") is not None
                ]
            records.append({
                "judge_hash": judge_hash,
                "cache_key": identifiers["cache_key"],
                "metric_name": self.metric_name,
                "question_index": chunk_idx,
                "judge_model": jm_config.get("model_id"),
                "judge_model_config": jm_config,
                "judge_input": {"pass": pass_name, "chunk_idx": chunk_idx},
                "question_score": None,
                "criteria": criteria,
                "raw_response": response,
                "error": None,
            })
        if records:
            cache_manager.save_rubric_outputs(records)

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
