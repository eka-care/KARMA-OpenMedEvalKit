"""
Base benchmark class with common functionality to reduce code redundancy.

This module provides a base class that handles:
- Cache management initialization and operations
- Statistics tracking and reporting
- Timing and performance measurement
- Model prediction patterns (supporting multimodal inputs)
- Weave integration
"""

import threading
import time
from typing import Any, Dict, List, Tuple, Optional

import weave
from torch.utils.data import DataLoader
from weave import EvaluationLogger

from karma.cache import CacheManager
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.models.base_model_abs import BaseHFModel

import logging

logger = logging.getLogger(__name__)


class Benchmark:
    def __init__(
        self,
        model: BaseHFModel,
        dataset: BaseMultimodalDataset,
        verbose_mode: bool = True,
        use_weave: bool = False,
        project_name: str = "benchmark-evaluation",
        enable_cache: bool = False,
        cache_path: str = "",
        console=None,
        progress=None,
        cache_manager: Optional[CacheManager] = None,
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
        """
        self.logger = logger
        self.model = model
        self.verbose_mode = verbose_mode
        self.progress = progress

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
        else:
            logger.info("No cache manager provided")

        self.dataset = dataset

    """
    Main benchmark class that works with any dataset and task combination.

    Handles cache management, statistics tracking, timing, and provides
    a generic evaluate function that works with any dataset/task pair.
    Supports multimodal inputs (text, images, audio, etc.)
    """

    def _create_weave_logger(self, dataset_name: str) -> EvaluationLogger:
        """
        Create Weave evaluation logger if enabled.

        Args:
            dataset_name: Dataset name for logging

        Returns:
            EvaluationLogger instance or None if disabled
        """
        # Sanitize model name for Weave (alphanumeric and underscores only)
        model_name = self.model.model_name_or_path
        sanitized_model_name = "".join(
            c if c.isalnum() or c == "_" else "_" for c in model_name
        )

        evaluation_logger = EvaluationLogger(
            name=sanitized_model_name, model=sanitized_model_name, dataset=dataset_name
        )
        self.logger.info("🔍 Weave EvaluationLogger initialized for summary tracking")
        return evaluation_logger

    def fetch_from_cache(self, samples: List[DataLoaderIterable]):
        """
        Fetch results from cache and return cache hits and misses.

        Args:
            samples: List of samples to look up in cache

        Returns:
            Tuple containing (cache_hits, samples_to_generate) where:
            - cache_hits: List of results found in cache
            - samples_to_generate: List of samples that need to be generated
        """
        results = []
        samples_to_generate = []
        # Step 1: Check cache for existing results
        cache_results = self.cache_manager.batch_fetch_rows(samples)
        cache_hits = 0

        for sample, cache_result in zip(samples, cache_results, strict=False):
            if cache_result:
                cache_hits += 1
                result = {
                    "prediction": cache_result.get("model_output", ""),
                    # "thinking_content": cache_result.get("model_output_reasoning", ""),
                    "from_cache": True,
                    "expected_output": sample.expected_output,
                }
                results.append(result)
            else:
                samples_to_generate.append(sample)

        if self.verbose_mode:
            self.logger.info(f"Cache hits: {cache_hits}/{len(samples)}")
            self.logger.info(
                f"Samples requiring generation: {len(samples_to_generate)}"
            )

        return results, samples_to_generate

    def batch_predict(
        self, samples: List[DataLoaderIterable], dry_run: bool = False
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
            start_time = time.time()

            batch_responses = self.model.run(inputs=samples)
            generation_time = time.time() - start_time
            self.logger.info(
                f"Model generation completed in {generation_time:.2f} seconds"
            )

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
                response = batch_response

                response = str(response)
                prediction, success = self.dataset.extract_prediction(response)

                result = {
                    "prediction": prediction,
                    "from_cache": False,
                    "sample": sample.model_dump(),
                    "expected_output": expected,
                    "success": success,
                }
                results.append(result)

                if self.progress and progress_task is not None:
                    self.progress.advance(progress_task)
            if self.progress and progress_task is not None:
                self.progress.remove_task(progress_task)

            # Step 3: Save new results to cache
            if self.enable_cache:
                self.logger.info(
                    f"Starting asynchronous cache update for datapoint {results}"
                )
                cache_thread = threading.Thread(
                    target=self.cache_manager.batch_save_rows,
                    args=(results,),
                    daemon=True,
                )
                cache_thread.start()
        elif dry_run:
            self.logger.info("Returning dummy results for dry run")
            # For dry run, return dummy results for cache misses
            for _, sample in enumerate(samples):
                expected = sample["expected_output"]

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

    def compute_metrics(
        self,
        prediction_results: List[Dict[str, Any]],
        metric_config: Dict[str, Any],
    ) -> Dict[str, float] | None:
        """
        Compute metrics for prediction results using multi-threading.

        Args:
            prediction_results: List of prediction result dictionaries
            metric_config: Configuration dictionary containing metric name and processors

        Returns:
            Dictionary of metric scores or None if metric not found
        """
        metric = metric_config["metric"]
        references = [
            self.dataset.postprocess(it["expected_output"]) for it in prediction_results
        ]
        predictions = [
            self.dataset.postprocess(it["prediction"]) for it in prediction_results
        ]
        score = metric.evaluate(predictions=predictions, references=references)
        return score

    def evaluate(
        self, metric_config: Dict[str, Any], batch_size: int = 1, dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Generic evaluate function that works with any dataset.

        Args:
            metric_config: Configuration dictionary containing metric name and processors
            batch_size: Batch size for evaluation
            dry_run: If True, only check cache status without running model inference

        Returns:
            Dictionary containing overall score, predictions, and summary data
        """
        if dry_run:
            self.logger.info(
                f"🔍 Starting DRY RUN with {self.dataset.__class__.__name__}"
            )
        else:
            self.logger.info(
                f"🚀 Starting evaluation with {self.dataset.__class__.__name__}"
            )
        self.logger.info(f"📦 Batch size: {batch_size}")

        start_time = time.time()

        # Initialize Weave EvaluationLogger
        evaluation_logger = None
        if self.use_weave:
            evaluation_logger = self._create_weave_logger(
                self.dataset.dataset_name.replace("/", "_")
            )

        if self.progress:
            print("here")
            task = self.progress.add_task(
                f"[cyan]Processing batches for {self.dataset.dataset_name}", total=None
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

        # Process batches from dataloader
        for batch_idx, samples in enumerate(dataloader):
            self.logger.info(batch_idx)
            if batch_idx > 2:
                break
            batch_results = []
            # samples = [
            #     dict(s) for s in samples
            # ]  # Ensure samples are proper dictionaries
            if self.enable_cache:
                batch_results, samples_to_generate = self.fetch_from_cache(samples)
            else:
                samples_to_generate = samples

            # Generate predictions using base class method
            if len(samples_to_generate) > 0:
                model_results = self.batch_predict(
                    samples=samples_to_generate, dry_run=dry_run
                )
                batch_results.extend(model_results)
            # Process results and extract answers using dataset template
            for result, sample in zip(batch_results, samples, strict=False):
                # Use dataset's extract_answer method (which uses template)
                expected = sample.expected_output

                # Create final prediction result
                prediction_result = {
                    "prediction": result["prediction"],
                    "expected_output": expected,
                    "sample": sample.model_dump(),
                    "from_cache": result.get("from_cache", False),
                    "success": result.get("success", True),
                }
                all_prediction_results.append(prediction_result)

                if self.verbose_mode:
                    self.logger.info(
                        f"Prediction: {result['prediction']}, Expected: {expected}, "
                        f"From cache: {result.get('from_cache', False)}"
                    )
            if self.progress and task is not None:
                self.progress.advance(task)

        if task:
            self.progress.remove_task(task)

        overall_score = self.compute_metrics(
            all_prediction_results, metric_config=metric_config
        )
        if overall_score is None:
            raise ValueError(
                f"Metric {metric_config['metric'].metric_name} not found in compute_metrics"
            )

        # Extract the main score from the metric results
        metric_name = metric_config["metric"].metric_name
        overall_score = overall_score[metric_name]
        # Create summary for Weave logging
        summary_data = {
            "overall_score": overall_score,
            "evaluation_time": time.time() - start_time,
        }

        # Log to Weave
        if self.use_weave and evaluation_logger is not None:
            evaluation_logger.log_summary(summary_data)
            self.logger.info(
                "📊 Summary results logged to Weave - check UI for comparisons!"
            )

        self.logger.info(f"\n🎯 Overall Score: {overall_score:.1%}")
        self.logger.info(f"⏱️  Total evaluation time: {time.time() - start_time:.2f}s")

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

        return {
            "overall_score": overall_score,
            "predictions": all_prediction_results,
            "summary": summary_data,
        }
