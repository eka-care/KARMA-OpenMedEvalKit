"""
Base benchmark class with common functionality to reduce code redundancy.

This module provides a base class that handles:
- Cache management initialization and operations
- Statistics tracking and reporting
- Timing and performance measurement
- Model prediction patterns (supporting multimodal inputs)
- Weave integration
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numbers

import weave
from torch.utils.data import DataLoader
from weave import EvaluationLogger

from karma.cache import CacheManager
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.models.base_model_abs import BaseModel
from karma.metrics.base_metric_abs import BaseMetric
import logging


_METRIC_DETAIL_VALUE_FIELDS: Dict[str, str] = {
    "rubric_evaluation": "question_score",
    "json_rubric_evaluation": "question_score",
}


def _extract_metric_value(metric_name: str, detail: Any) -> Optional[float]:
    """Extract a scalar value from a per-sample metric payload for persistence."""

    if isinstance(detail, numbers.Number) and not isinstance(detail, bool):
        return float(detail)

    if not isinstance(detail, dict):
        return None

    preferred_key = _METRIC_DETAIL_VALUE_FIELDS.get(metric_name)
    candidate_keys = []
    if preferred_key:
        candidate_keys.append(preferred_key)
    candidate_keys.extend(["question_score", "value", "score"])

    for key in candidate_keys:
        if key not in detail:
            continue
        candidate = detail.get(key)
        if isinstance(candidate, numbers.Number) and not isinstance(candidate, bool):
            return float(candidate)

    return None

logger = logging.getLogger(__name__)


class Benchmark:
    def __init__(
        self,
        model: BaseModel,
        dataset: BaseMultimodalDataset,
        verbose_mode: bool = False,
        use_weave: bool = False,
        project_name: str = "benchmark-evaluation",
        progress=None,
        cache_manager: Optional[CacheManager] = None,
        refresh_cache: bool = False,
        cached_only: bool = False,
    ):
        """
        Initialize benchmark for any dataset/task combination.

        Args:
            model: The language model to evaluate
            verbose_mode: Whether to self.logger.info detailed logs
            use_weave: Whether to use Weave EvaluationLogger
            project_name: Weave project name for tracking
            enable_cache: Whether to enable persistent caching
            cache_path: Path to cache database (used only if cache_manager is None)
            cache_manager: Optional pre-initialized CacheManager instance
            console: Optional rich Console for output (from orchestrator)
            progress: Optional rich Progress instance for progress bars (from orchestrator)
            refresh_cache: Whether to skip cache lookup and force regeneration
        """
        self.logger = logger
        self.model = model
        self.verbose_mode = verbose_mode
        self.progress = progress
        self.refresh_cache = refresh_cache
        # cached_only: score only samples that already have cached model outputs;
        # never run the (possibly GPU-bound) model. The metric/judge still runs.
        self.cached_only = cached_only

        if self.verbose_mode:
            self.logger.info(f"Initializing benchmark with model: {model}")
            self.logger.info(f"Dataset: {dataset.dataset_name}")

        # Weave integration
        self.use_weave = use_weave
        if self.use_weave:
            weave.init(project_name)
            self.logger.success(f"✅ Initialized Weave tracking: {project_name}")

        # Setup caching system
        self.enable_cache = cache_manager is not None

        self.cache_manager = cache_manager
        if self.cache_manager:
            logger.info("Cache manager initialized successfully")
            raw_subset = (
                getattr(dataset, "config", None)
                or getattr(dataset, "subset", None)
                or getattr(dataset, "language", None)
            )
            dataset_subset_label = (
                str(raw_subset).strip()
                if raw_subset is not None and str(raw_subset).strip()
                else None
            )
            self.cache_manager.record_dataset_metadata(
                dataset_subset=dataset_subset_label,
                split=getattr(dataset, "split", None),
            )
        else:
            logger.info("No cache manager provided")

        self.dataset = dataset

    """
    Main benchmark class that works with any dataset and task combination.

    Handles cache management, statistics tracking, timing, and provides
    a generic evaluate function that works with any dataset/task pair.
    Supports multimodal inputs (text, images, audio, etc.)
    """

    def _create_weave_logger(self, dataset_name: str):
        """
        Create Weave evaluation logger if enabled.

        Args:
            dataset_name: Dataset name for logging

        Returns:
            EvaluationLogger instance or None if disabled
        """
        # Sanitize model name for Weave (alphanumeric and underscores only)
        model_name = getattr(self.model, "model_name_or_path", "cached") if self.model is not None else "cached"
        sanitized_model_name = "".join(
            c if c.isalnum() or c == "_" else "_" for c in model_name
        )

        evaluation_logger = EvaluationLogger(
            name=sanitized_model_name, model=sanitized_model_name, dataset=dataset_name
        )
        self.logger.info("🔍 Weave EvaluationLogger initialized for summary tracking")
        return evaluation_logger

    def _standardize_sample_languages(self, samples: List[DataLoaderIterable]) -> None:
        """Enforce the language contract before hashing and inference:
        single-language datasets force their config language onto EVERY sample
        (including unknown-language rows — one subset, one cache-key language);
        multilingual datasets normalize the row value. A language that cannot
        be normalized to an ISO code fails the eval — unmapped languages must
        be added to karma.utils.language, never hashed raw."""
        from karma.utils.language import normalize_language

        config_language = normalize_language(getattr(self.dataset, "language", None))
        for sample in samples:
            if config_language:
                sample.language = config_language
            elif sample.language is not None:
                sample.language = normalize_language(sample.language)

    def _extract_with_taxonomy(
        self, raw_output: str, sample: DataLoaderIterable, model_error: bool = False
    ):
        """Spec B3/B4: model_error = the model call itself failed (error set);
        a blank response is a valid success. An extraction failure or exception
        marks that one sample post_processing_error, never the batch."""
        try:
            prediction, extracted_ok = self.dataset.extract_prediction(
                raw_output, other_args=sample.other_args
            )
        except Exception:
            self.logger.warning(
                "extract_prediction failed on a sample; marking post_processing_error",
                exc_info=True,
            )
            prediction, extracted_ok = "", False
        if model_error:
            return prediction, False, "model_error"
        if extracted_ok:
            return prediction, True, None
        return prediction, False, "post_processing_error"

    def fetch_from_cache(self, samples: List[DataLoaderIterable]):
        """
        Fetch results from cache and return cache hits and misses.

        Args:
            samples: List of samples to look up in cache

        Returns:
            Tuple containing (results, samples_to_generate, indices_to_generate) where:
            - results: List of results with None placeholders for cache misses
            - samples_to_generate: List of samples that need to be generated
            - indices_to_generate: List of indices where generated results should be placed
        """
        results = []
        samples_to_generate = []
        indices_to_generate = []

        # Skip cache lookup if refresh_cache is True
        if self.refresh_cache:
            if self.verbose_mode:
                self.logger.info(
                    f"Cache refresh enabled - skipping cache lookup for {len(samples)} samples"
                )
            return [None] * len(samples), samples, list(range(len(samples)))

        # Step 1: Check cache for existing results
        cache_results = self.cache_manager.batch_fetch_rows(samples)
        cache_hits = 0

        for idx, (sample, cache_result) in enumerate(zip(samples, cache_results, strict=False)):
            if cache_result:
                cache_hits += 1
                # Re-extract from raw (C10); NULL raw = legacy row, use processed as-is.
                raw_output = cache_result.get("raw_output")
                if raw_output is not None:
                    prediction, success, error_type = self._extract_with_taxonomy(
                        raw_output, sample
                    )
                else:
                    prediction = cache_result.get("processed_output")
                    if prediction is None:
                        prediction = cache_result.get("model_output", "")
                    success = bool(cache_result.get("success", True))
                    error_type = cache_result.get("error_type")
                result = {
                    "prediction": prediction,
                    "tool_trace": cache_result.get("tool_trace", "") or "",
                    "sample": sample,
                    "sample_serialized": sample.model_dump(),
                    # "thinking_content": cache_result.get("model_output_reasoning", ""),
                    "from_cache": True,
                    "expected_output": sample.expected_output,
                    "success": success,
                    "error_type": error_type,
                }
                results.append(result)
            else:
                results.append(None)  # Placeholder for cache miss
                samples_to_generate.append(sample)
                indices_to_generate.append(idx)

        if self.verbose_mode:
            self.logger.info(f"Cache hits: {cache_hits}/{len(samples)}")
            self.logger.info(
                f"Samples requiring generation: {len(samples_to_generate)}"
            )

        return results, samples_to_generate, indices_to_generate

    def batch_predict(
        self,
        samples: List[DataLoaderIterable],
        dry_run: bool = False,
        dataset_name=None,
    ) -> List[Dict[str, Any]]:
        """
        Generate predictions for a batch of samples, with cache checking and fetching.
        """
        n_samples = len(samples)
        self.logger.debug(f"Starting batch prediction for {n_samples} samples")
        if dry_run:
            self.logger.debug("Dry run mode enabled - skipping model inference")

        results = []

        if not dry_run:
            self.logger.debug("Generating batch responses from model")
            # start_time = time.time()

            batch_responses = self.model.run_with_metadata(inputs=samples)
            # generation_time = time.time() - start_time
            # self.logger.info(
            #     f"Model generation completed in {generation_time:.2f} seconds"
            # )

            # Use progress bar if available for per-sample progress
            progress_task = None
            if self.progress:
                progress_task = self.progress.add_task(
                    "Batch prediction", total=n_samples
                )

            for i, (batch_response, sample) in enumerate(
                zip(batch_responses, samples, strict=False)
            ):
                expected = sample.expected_output
                response = str(batch_response.get("text", ""))
                tool_trace = batch_response.get("tool_trace", "") or ""
                err = batch_response.get("error")
                prediction, success, error_type = self._extract_with_taxonomy(
                    response, sample, model_error=bool(err)
                )

                result = {
                    "prediction": prediction,
                    "tool_trace": tool_trace,
                    "raw_output": response,
                    "from_cache": False,
                    "sample": sample.model_dump(),
                    "expected_output": expected,
                    "success": success,
                    "error_type": error_type,
                }
                results.append(result)

                if self.progress and progress_task is not None:
                    self.progress.advance(progress_task)
            if self.progress and progress_task is not None:
                self.progress.remove_task(progress_task)

            # Step 3: Save new results to cache
            if self.enable_cache:
                # self.logger.info(
                #     f"Starting asynchronous cache update for datapoint"
                # )
                cache_thread = threading.Thread(
                    target=self.cache_manager.batch_save_rows,
                    args=(
                        results,
                        dataset_name,
                    ),
                    daemon=False,
                )
                cache_thread.start()
                cache_thread.join(timeout=120)
        elif dry_run:
            self.logger.info("Returning dummy results for dry run")
            # For dry run, return dummy results for cache misses
            for _, sample in enumerate(samples):
                expected = sample.expected_output

                result = {
                    "prediction": "DRY_RUN",
                    "thinking_content": "",
                    "from_cache": False,
                    "sample": sample.model_dump(),
                    "expected_output": expected,
                    "success": True,
                }
                results.append(result)

        return results

    def _replay_gap_beyond_known_errors(self, gap: int) -> bool:
        """Gap beyond a completed run's model errors = crashed-run holes → dev.
        Unknown history counts as unexplained."""
        getter = getattr(
            self.cache_manager.cache_io, "get_completed_run_error_count", None
        )
        if not callable(getter):
            return True
        try:
            known_errors = getter(self.cache_manager.config_hash)
        except Exception:
            self.logger.exception(
                "Could not fetch completed-run error count; treating replay as partial"
            )
            return True
        return known_errors is None or gap > int(known_errors)

    def compute_metrics(
        self,
        ground_truth_and_prediction: List[Dict[str, Any]],
        metrics: List[BaseMetric],
        *,
        persist: bool = True,
    ) -> Dict[str, float] | None:
        """
        Compute metrics for prediction results using multi-threading.

        Args:
            ground_truth_and_prediction: List of prediction result dictionaries
            metric_config: Configuration dictionary containing metric name and processors
            persist: When False, skips writing metric summaries to cache

        Returns:
            Dictionary of metric scores or None if metric not found
        """
        # Failed samples never reach metric math (spec 3.2): model errors and
        # post-processing errors are excluded here — the ONE filtering place.
        scored_items: List[Dict[str, Any]] = []
        model_error_count = 0
        post_processing_error_count = 0
        for it in ground_truth_and_prediction:
            success = bool(it.get("success", True))
            error_type = it.get("error_type") or (None if success else "model_error")
            if error_type == "model_error":
                model_error_count += 1
            elif error_type == "post_processing_error":
                post_processing_error_count += 1
            else:
                scored_items.append(it)
        self.metric_error_counts = {
            "model_error": model_error_count,
            "post_processing_error": post_processing_error_count,
        }
        if model_error_count or post_processing_error_count:
            self.logger.info(
                "Metrics computed over %d of %d samples (excluded: %d model errors, "
                "%d post-processing errors)",
                len(scored_items),
                len(ground_truth_and_prediction),
                model_error_count,
                post_processing_error_count,
            )

        scores = {}
        references = []
        predictions = []
        rubrics = []
        samples = []
        entities = []
        qualities = []
        successes: List[bool] = []
        serialized_samples: List[Any] = []
        for it in scored_items:
            predictions.append(it["prediction"])
            references.append(it["expected_output"])
            entities.append(it["entities"])
            qualities.append(it.get("quality"))
            successes.append(bool(it.get("success", True)))
            if it.get("sample"):
                samples.append(it["sample"])
                rubrics.append(it["sample"].rubric_to_evaluate)
            serialized_samples.append(it.get("sample_serialized"))
        predictions = self.dataset.postprocess(predictions)
        references = self.dataset.postprocess(references)

        total_samples = len(ground_truth_and_prediction)
        metric_records: List[Dict[str, Any]] = []
        if persist and self.enable_cache and self.cache_manager:
            sample_identifiers: List[Optional[Dict[str, str]]] = [None] * len(serialized_samples)
        else:
            sample_identifiers = []

        # Metric hint only — lenient by design; hash strictness lives in the chokepoint.
        from karma.utils.language import metric_language

        language = metric_language(getattr(self.dataset, "language", None))

        def evaluate_single_metric(metric: BaseMetric) -> Dict[str, Any]:
            """Evaluate a single metric and return its results."""
            score = metric.evaluate(
                predictions=predictions,
                references=references,
                language=language,
                rubrics=rubrics,
                samples=samples,
                entities=entities,
                qualities=qualities,
                successes=successes,
                serialized_samples=serialized_samples,
                cache_manager=self.cache_manager,
                dataset_name=self.dataset.dataset_name,
            )
            return {"metric": metric, "score": score}

        # Parallelize metric evaluation using ThreadPoolExecutor
        metric_results = []
        metric_names = [m.metric_name for m in metrics]
        self.logger.info(
            "⏳ Computing %d metrics over %d samples: %s",
            len(metrics),
            total_samples,
            ", ".join(metric_names),
        )
        metrics_start = time.time()
        completed = 0
        with ThreadPoolExecutor(max_workers=len(metrics)) as executor:
            future_to_metric = {
                executor.submit(evaluate_single_metric, metric): metric
                for metric in metrics
            }
            for future in as_completed(future_to_metric):
                metric = future_to_metric[future]
                try:
                    result = future.result()
                    metric_results.append(result)
                    completed += 1
                    self.logger.info(
                        "✅ Metric %s done (%d/%d, %.1fs elapsed)",
                        metric.metric_name,
                        completed,
                        len(metrics),
                        time.time() - metrics_start,
                    )
                except Exception as e:
                    self.logger.exception(
                        f"Failed to evaluate metric {metric.metric_name}: {e}"
                    )

        # Process results from parallel evaluation
        for metric_result in metric_results:
            metric = metric_result["metric"]
            score = metric_result["score"]

            detail_key = f"{metric.metric_name}_details"
            per_sample_details = None
            if isinstance(score, dict):
                metric_value = score.get(metric.metric_name)
                scores[metric.metric_name] = metric_value

                per_sample_details = score.get(detail_key)
                if per_sample_details:
                    for result, detail in zip(
                        scored_items,
                        per_sample_details,
                        strict=False,
                    ):
                        metrics_payload = result.setdefault("metrics", {})
                        metrics_payload[metric.metric_name] = detail

                run_payload = {
                    key: value for key, value in score.items() if key != detail_key
                }
                if metric.non_percentage_fields:
                    run_payload["_non_percentage_fields"] = sorted(
                        metric.non_percentage_fields
                    )
            else:
                scores[metric.metric_name] = score
                metric_value = score
                run_payload = None

            field_support = None
            overall_value = None
            if isinstance(run_payload, dict):
                field_support = run_payload.pop("_field_support", None)
                overall_value = run_payload.pop("_overall_value", None)

            should_persist = (
                persist and self.enable_cache and self.cache_manager is not None
            )

            if should_persist:
                scored_count: Optional[int]
                if isinstance(per_sample_details, list):
                    scored_count = sum(1 for detail in per_sample_details if detail is not None)
                else:
                    scored_count = len(scored_items)

                # metric_results.value is scalar; dict-valued breakdown metrics
                # store their overall value (or NULL) + explode per-field below.
                parent_value = (
                    metric_value
                    if isinstance(metric_value, (int, float)) or metric_value is None
                    else overall_value
                )
                metric_records.append(
                    {
                        "metric_name": metric.metric_name,
                        "target_scope": "run",
                        "target_id": "__run__",
                        "value": parent_value,
                        "payload": run_payload,
                        "total_count": total_samples,
                        "scored_count": scored_count,
                    }
                )

                # Per-field explode (spec 3.3): one run-scope row per breakdown
                # field with its own support, e.g. "Quality_high_Semantic_WER".
                if isinstance(field_support, dict):
                    for field_name, support in field_support.items():
                        if not isinstance(support, dict):
                            continue
                        metric_records.append(
                            {
                                "metric_name": field_name,
                                "target_scope": "run",
                                "target_id": "__run__",
                                "value": support.get("value"),
                                "payload": None,
                                "total_count": support.get("total"),
                                "scored_count": support.get("scored"),
                            }
                        )

                if isinstance(per_sample_details, list) and sample_identifiers:
                    for idx, detail in enumerate(per_sample_details):
                        if detail is None or idx >= len(serialized_samples):
                            continue

                        serialized_sample = serialized_samples[idx]
                        if serialized_sample is None:
                            continue

                        identifiers = sample_identifiers[idx]
                        if identifiers is None:
                            try:
                                identifiers = self.cache_manager.get_cache_identifiers_for_sample(
                                    serialized_sample
                                )
                            except Exception:  # pragma: no cover - hashing failures should not abort
                                self.logger.exception(
                                    "Failed to derive cache identifiers for metric persistence"
                                )
                                sample_identifiers[idx] = None
                                continue
                            else:
                                sample_identifiers[idx] = identifiers

                        if identifiers is None:
                            continue

                        metric_records.append(
                            {
                                "metric_name": metric.metric_name,
                                "target_scope": "sample",
                                "target_id": identifiers["dataset_row_hash"],
                                "cache_key": identifiers["cache_key"],
                                "value": _extract_metric_value(
                                    metric.metric_name, detail
                                ),
                                "payload": detail,
                            }
                        )

        if metric_records and persist and self.enable_cache and self.cache_manager:
            try:
                self.cache_manager.save_metric_results(metric_records)
            except Exception:  # pragma: no cover - persistence should not break evaluation
                self.logger.exception("Failed to persist metric results")

        return scores

    def evaluate(
        self,
        metrics: List[BaseMetric],
        batch_size: int = 1,
        dry_run: bool = False,
        refresh_cache: Optional[bool] = False,
    ) -> Dict[str, Any]:
        """
        Generic evaluate function that works with any dataset.

        Args:
            metrics:
            metric_config: Configuration dictionary containing metric name and processors
            batch_size: Batch size for evaluation
            dry_run: If True, only check cache status without running model inference
            refresh_cache: If True, skip cache lookup and force regeneration (overrides instance setting)

        Returns:
            Dictionary containing overall score, predictions, and summary data
        """
        # Override instance refresh_cache setting if parameter is provided
        if refresh_cache:
            original_refresh_cache = self.refresh_cache
            self.refresh_cache = refresh_cache

        if dry_run:
            self.logger.info(
                f"🔍 Starting DRY RUN with {self.dataset.__class__.__name__}"
            )
        else:
            self.logger.info(
                f"🚀 Starting evaluation with {self.dataset.__class__.__name__}"
            )
        # self.logger.info(f"📦 Batch size: {batch_size}")

        start_time = time.time()

        # Initialize Weave EvaluationLogger
        evaluation_logger = None
        if self.use_weave:
            evaluation_logger = self._create_weave_logger(
                self.dataset.dataset_name.replace("/", "_")
            )

        if self.progress:
            try:
                total_samples = len(self.dataset)
            except TypeError:
                total_samples = None
            task = self.progress.add_task(
                f"[cyan]Processing samples for {self.dataset.dataset_name}",
                total=total_samples,
            )
        else:
            task = None

        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
            # prefetch_factor=2,
            # num_workers=1,
            collate_fn=self.dataset.collate_fn,
            drop_last=False,
        )

        all_prediction_results = []
        skipped_no_cache = 0

        # Process batches from dataloader
        for batch_idx, samples in enumerate(dataloader):
            self._standardize_sample_languages(samples)
            batch_results = [None] * len(samples)
            indices_to_generate = []
            # samples = [
            #     dict(s) for s in samples
            # ]  # Ensure samples are proper dictionaries
            if self.enable_cache:
                batch_results, samples_to_generate, indices_to_generate = self.fetch_from_cache(samples)
            else:
                samples_to_generate = samples
                indices_to_generate = list(range(len(samples)))

            # cached_only: never invoke the model — evaluate only what is cached.
            if self.cached_only:
                samples_to_generate = []

            # Generate predictions using base class method
            if len(samples_to_generate) > 0:
                model_results = self.batch_predict(
                    samples=samples_to_generate,
                    dry_run=dry_run,
                    dataset_name=self.dataset.dataset_name,
                )
                # Place generated results at their correct indices
                for idx, result in zip(indices_to_generate, model_results):
                    batch_results[idx] = result
            # Process results and extract answers using dataset template
            for result, sample in zip(batch_results, samples, strict=False):
                # cached_only: a None slot is a sample with no cached output — skip it.
                if result is None:
                    skipped_no_cache += 1
                    continue
                # Use dataset's extract_answer method (which uses template)
                expected = sample.expected_output
                if sample.other_args:
                    entities = sample.other_args.get("entities")
                    quality = sample.other_args.get("quality")
                else:
                    entities = None
                    quality = None

                serialized_sample = result.get("sample_serialized")
                if serialized_sample is None:
                    serialized_sample = sample.model_dump()

                # Create final prediction result
                prediction_result = {
                    "prediction": result["prediction"],
                    "tool_trace": result.get("tool_trace", ""),
                    "expected_output": expected,
                    "entities": entities,
                    "quality": quality,
                    "sample": sample,
                    "sample_serialized": serialized_sample,
                    "from_cache": result.get("from_cache", False),
                    "success": result.get("success", True),
                    "error_type": result.get("error_type"),
                }
                all_prediction_results.append(prediction_result)

                if self.verbose_mode:
                    self.logger.info(
                        f"Prediction: {result['prediction']}, Expected: {expected}, "
                        f"From cache: {result.get('from_cache', False)}"
                    )
            if self.progress and task is not None:
                self.progress.advance(task, advance=len(samples))

            if self.enable_cache and self.cache_manager is not None:
                self.logger.info(
                    "📋 Processed %d samples | cache_hits=%d misses=%d (%s)",
                    len(all_prediction_results),
                    self.cache_manager.database_hits,
                    self.cache_manager.database_misses,
                    self.dataset.dataset_name,
                )
            else:
                self.logger.info(
                    "📋 Processed %d samples (%s)",
                    len(all_prediction_results),
                    self.dataset.dataset_name,
                )

        if task:
            self.progress.remove_task(task)

        if self.cached_only:
            cfg = self.cache_manager.config_hash if self.cache_manager else "?"
            if not all_prediction_results:
                msg = (
                    f"No cached model outputs found for config_hash={cfg} "
                    f"(dataset={self.dataset.dataset_name}). Nothing to evaluate — "
                    f"run without --cached-only first to generate the model outputs."
                )
                print(msg)
                self.logger.warning(msg)
                return {
                    "overall_score": {},
                    "predictions": [],
                    "summary": {
                        "evaluation_time": time.time() - start_time,
                        "no_cache": True,
                    },
                }
            self.logger.info(
                "cached-only: scoring %d cached samples (skipped %d with no cache); "
                "model was not loaded.",
                len(all_prediction_results), skipped_no_cache,
            )

        overall_scores = self.compute_metrics(
            all_prediction_results,
            metrics=metrics,
            persist=not dry_run,
        )

        # Create summary for Weave logging
        summary_data = {
            "overall_score": overall_scores,
            "evaluation_time": time.time() - start_time,
        }

        # Log to Weave
        if self.use_weave and evaluation_logger is not None:
            evaluation_logger.log_summary(summary_data)
            self.logger.info(
                "📊 Summary results logged to Weave - check UI for comparisons!"
            )

        # self.logger.info(f"\n🎯 Overall Score: {overall_scores}")
        # self.logger.info(f"⏱️  Total evaluation time: {time.time() - start_time:.2f}s")

        if self.enable_cache:
            hit_rate = (
                self.cache_manager.database_hits
                / (
                    self.cache_manager.database_hits
                    + self.cache_manager.database_misses
                )
                * 100
                if (
                    self.cache_manager.database_hits
                    + self.cache_manager.database_misses
                )
                > 0
                else 0.0
            )
            self.logger.info(
                f"🗂️  Cache hits: {self.cache_manager.database_hits}, "
                f"misses: {self.cache_manager.database_misses}, "
                f"hit rate: {hit_rate:.1f}%"
            )
            empty_outputs = sum(
                1 for r in all_prediction_results
                if not str(r.get("prediction") or "").strip()
            )
            self.logger.info(
                "📋 Run summary | dataset=%s total=%d empty=%d cache_hits=%d "
                "misses=%d hit_rate=%.1f%% time=%.1fs",
                self.dataset.dataset_name,
                len(all_prediction_results),
                empty_outputs,
                self.cache_manager.database_hits,
                self.cache_manager.database_misses,
                hit_rate,
                time.time() - start_time,
            )

            if dry_run:
                self.logger.info("\n" + "=" * 40)
                self.logger.info("🎯 DRY RUN COMPLETE")
                self.logger.info(f"📊 Cache hit rate: {hit_rate:.1f}%")
                self.logger.info(
                    f"✅ {self.cache_manager.database_hits} samples in cache"
                )
                self.logger.info(
                    f"🔄 {self.cache_manager.database_misses} samples would need inference"
                )

        # Restore original refresh_cache setting if it was overridden
        if refresh_cache:
            self.refresh_cache = original_refresh_cache

        if (
            self.enable_cache
            and self.cache_manager is not None
            and not dry_run
        ):
            total = len(all_prediction_results)
            model_errors = 0
            post_processing_errors = 0
            for r in all_prediction_results:
                if bool(r.get("success", True)):
                    continue
                if r.get("error_type") == "post_processing_error":
                    post_processing_errors += 1
                else:
                    model_errors += 1
            processed = total - model_errors - post_processing_errors
            # cached_only: count skipped samples into total = real coverage.
            replay_gap = skipped_no_cache if self.cached_only else 0
            total += replay_gap
            updater = getattr(self.cache_manager.cache_io, "update_run_counts", None)
            if callable(updater):
                try:
                    updater(
                        self.cache_manager.config_hash,
                        total=total,
                        processed=processed,
                        error=model_errors,
                        job_id=getattr(self.cache_manager, "job_id", None),
                        post_processing_errors=post_processing_errors,
                        cache_hits=self.cache_manager.database_hits,
                        duration_seconds=time.time() - start_time,
                    )
                except Exception:
                    self.logger.exception(
                        "Failed to update run counts after evaluation; "
                        "UI totals may be stale until the next write."
                    )
            # Completed — unless every sample errored (dead key/outage): a run
            final_status = "completed" if (processed > 0 or total == 0) else "failed"
            try:
                payload = {
                    "job_id": getattr(self.cache_manager, "job_id", None),
                    "config_hash": self.cache_manager.config_hash,
                    "status": final_status,
                }
                if final_status == "failed":
                    payload["error_message"] = (
                        f"all {total} samples errored (model_errors={model_errors}, "
                        f"post_processing_errors={post_processing_errors})"
                    )
                if replay_gap and self._replay_gap_beyond_known_errors(replay_gap):
                    payload["run_mode"] = "dev"
                self.cache_manager.cache_io.save_run_metadata(payload)
            except Exception:
                self.logger.exception(
                    "Failed to mark run status as completed after evaluation"
                )

        if (
            self.enable_cache
            and self.cache_manager is not None
            and not dry_run
            and os.environ.get("KARMA_CONFIG_HASH", "").strip()
        ):
            remover = getattr(self.cache_manager.cache_io, "remove_run_status", None)
            if callable(remover):
                try:
                    remover(self.cache_manager.config_hash)
                except Exception: 
                    self.logger.exception("Failed to clear run status after evaluation")

        return {
            "overall_score": overall_scores,
            "predictions": all_prediction_results,
            "summary": summary_data,
        }
