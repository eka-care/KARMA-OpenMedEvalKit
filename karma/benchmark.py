"""
Base benchmark class with common functionality to reduce code redundancy.

This module provides a base class that handles:
- Cache management initialization and operations
- Statistics tracking and reporting
- Timing and performance measurement
- Model prediction patterns (supporting multimodal inputs)
- Weave integration
"""

import logging
import threading
import time
from typing import Any, Dict, List, Tuple

import weave
from torch.utils.data import DataLoader
from weave import EvaluationLogger

from karma.cache import CacheManager
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.models.base import BaseLLM

import evaluate


class Benchmark:
    """
    Main benchmark class that works with any dataset and task combination.

    Handles cache management, statistics tracking, timing, and provides
    a generic evaluate function that works with any dataset/task pair.
    Supports multimodal inputs (text, images, audio, etc.)
    """

    def __init__(
        self,
        logger,
        model: BaseLLM,
        dataset: BaseMultimodalDataset,
        verbose_mode: bool = False,
        use_weave: bool = False,
        project_name: str = "benchmark-evaluation",
        enable_cache: bool = False,
        cache_path: str = "",
    ):
        """
        Initialize benchmark for any dataset/task combination.

        Args:
            model: The language model to evaluate
            verbose_mode: Whether to self.logger.info detailed logs
            use_weave: Whether to use Weave EvaluationLogger
            project_name: Weave project name for tracking
            enable_cache: Whether to enable persistent caching
        """
        # Initialize DeepEval base
        # Store model and basic settings
        self.logger = logger
        self.model = model
        self.verbose_mode = verbose_mode

        logger.info(f"Initializing benchmark with model: {model.model_config.model_id}")
        logger.info(f"Dataset: {dataset.dataset_name}")

        # Weave integration
        self.use_weave = use_weave
        if self.use_weave:
            weave.init(project_name)
            logger.info(f"✅ Initialized Weave tracking: {project_name}")

        # Setup caching system
        self.enable_cache = enable_cache
        if self.enable_cache:
            logger.info(f"Enabling cache with path: {cache_path}")
            self.cache_manager = CacheManager(
                dataset.dataset_name, self.model.model_config, cache_path=cache_path
            )
            logger.info("Cache manager initialized successfully")
        self.dataset = dataset

    def get_batch_model_inputs(
        self, samples: List[Dict[str, Any]]
    ) -> Dict[str, List[Any]]:
        """Convert batch of samples to multimodal model inputs."""
        batch_inputs = {
            "prompts": [sample["input"] for sample in samples],
            "images": [sample.get("images", None) for sample in samples],
            "audios": [sample.get("audios", None) for sample in samples],
        }

        # Remove keys where all values are None
        batch_inputs = {
            k: v for k, v in batch_inputs.items() if any(x is not None for x in v)
        }

        return batch_inputs

    def _create_weave_logger(self, dataset_name: str) -> EvaluationLogger:
        """
        Create Weave evaluation logger if enabled.

        Args:
            dataset_name: Dataset name for logging

        Returns:
            EvaluationLogger instance or None if disabled
        """
        # Sanitize model name for Weave (alphanumeric and underscores only)
        model_name = self.model.model_config.model_id
        sanitized_model_name = "".join(
            c if c.isalnum() or c == "_" else "_" for c in model_name
        )

        evaluation_logger = EvaluationLogger(
            name=sanitized_model_name, model=sanitized_model_name, dataset=dataset_name
        )
        self.logger.info("🔍 Weave EvaluationLogger initialized for summary tracking")
        return evaluation_logger

    def fetch_from_cache(
        self, samples: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Fetch results from cache and return cache hits and misses.

        Args:
            samples: List of samples to look up in cache

        Returns:
            Tuple containing (cache_hits, samples_to_generate) where:
            - cache_hits: List of results found in cache
            - samples_to_generate: List of samples that need to be generated
        """
        if self.verbose_mode:
            self.logger.info(f"Attempting to fetch {len(samples)} samples from cache")

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
                    "thinking_content": cache_result.get("model_output_reasoning", ""),
                    "from_cache": True,
                    "expected_output": sample.get("expected_output", ""),
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
        self, samples: List[Dict[str, Any]], dry_run: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate predictions for a batch of samples, with cache checking and fetching.
        """
        self.logger.info(f"Starting batch prediction for {len(samples)} samples")
        if dry_run:
            self.logger.info("Dry run mode enabled - skipping model inference")

        results = []

        if not dry_run:
            model_inputs_to_generate = self.get_batch_model_inputs(samples)
            self.logger.info("Generating batch responses from model")
            start_time = time.time()

            # Use batch generation with model inputs passed as kwargs
            batch_responses = self.model.batch_generate(**model_inputs_to_generate)

            generation_time = time.time() - start_time
            self.logger.info(
                f"Model generation completed in {generation_time:.2f} seconds"
            )

            for i, (batch_response, sample) in enumerate(
                zip(batch_responses, samples, strict=False)
            ):
                expected = sample["expected_output"]

                # batch_generate returns (thinking_content, response) tuples
                if (
                    isinstance(batch_response, (tuple, list))
                    and len(batch_response) == 2
                ):
                    thinking_content, response = batch_response
                else:
                    # Fallback for unexpected format
                    thinking_content = ""
                    response = batch_response

                response = str(response)
                thinking_content = str(thinking_content)
                prediction, success = self.dataset.extract_prediction(response)
                result = {
                    "prediction": prediction,
                    "thinking_content": thinking_content,
                    "from_cache": False,
                    "sample": sample,
                    "expected_output": expected,
                    "success": success,
                }
                results.append(result)

            # Step 3: Save new results to cache
            if self.enable_cache:
                self.logger.info("Starting asynchronous cache update")
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
                    "sample": sample,
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
        references = [it["expected_output"] for it in prediction_results]
        predictions = [it["prediction"] for it in prediction_results]
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

        # Create dataloader
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
            samples = [
                dict(s) for s in samples
            ]  # Ensure samples are proper dictionaries
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
                expected = sample["expected_output"]

                # Create final prediction result
                prediction_result = {
                    "prediction": result["prediction"],
                    "expected_output": expected,
                    "sample": sample,
                    "from_cache": result.get("from_cache", False),
                    "success": result.get("success", True),
                }
                all_prediction_results.append(prediction_result)

                if self.verbose_mode:
                    self.logger.info(
                        f"Prediction: {result['prediction']}, Expected: {expected}, "
                        f"From cache: {result.get('from_cache', False)}"
                    )
        # Calculate overall results
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
