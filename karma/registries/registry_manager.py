"""
Central registry manager for coordinated discovery across all registries.

This module provides a unified interface for discovering all registries
in parallel, significantly improving CLI startup performance.
"""

import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RegistryDiscoveryResult:
    """Result of a registry discovery operation."""

    registry_name: str
    success: bool
    discovery_time: float
    error: Optional[str] = None
    from_cache: bool = False


class RegistryManager:
    """
    Central manager for coordinating registry discovery operations.

    Provides parallel discovery, caching coordination, and performance monitoring
    for all karma registries (model, dataset, metrics, processor).
    """

    def __init__(self):
        self._discovery_lock = threading.RLock()
        self._discovery_results: Dict[str, RegistryDiscoveryResult] = {}
        self._total_discovery_time = 0.0

    def discover_all_registries(
        self, use_cache: bool = True, parallel: bool = True
    ) -> Dict[str, RegistryDiscoveryResult]:
        """
        Discover all registries with optional parallel execution.

        Args:
            use_cache: Whether to use cached discovery results
            parallel: Whether to run discoveries in parallel

        Returns:
            Dictionary of discovery results for each registry
        """
        with self._discovery_lock:
            start_time = time.time()

            if parallel:
                results = self._discover_parallel(use_cache)
            else:
                results = self._discover_sequential(use_cache)

            self._total_discovery_time = time.time() - start_time
            self._discovery_results = results

            # Log summary
            cache_hits = sum(1 for r in results.values() if r.from_cache)
            successful = sum(1 for r in results.values() if r.success)

            logger.info(
                f"Registry discovery completed: {successful}/{len(results)} successful, "
                f"{cache_hits} cache hits, total time: {self._total_discovery_time:.2f}s"
            )

            return results

    def _discover_parallel(self, use_cache: bool) -> Dict[str, RegistryDiscoveryResult]:
        """Discover registries in parallel using ThreadPoolExecutor."""
        results = {}

        # Define discovery functions for each registry
        discovery_tasks = [
            ("model", self._discover_model_registry),
            ("dataset", self._discover_dataset_registry),
            ("metrics", self._discover_metrics_registry),
            ("processor", self._discover_processor_registry),
        ]

        logger.debug(
            f"Starting parallel discovery of {len(discovery_tasks)} registries"
        )

        with ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="registry-discovery"
        ) as executor:
            # Submit all discovery tasks
            future_to_registry = {
                executor.submit(discover_func, use_cache): registry_name
                for registry_name, discover_func in discovery_tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_registry):
                registry_name = future_to_registry[future]
                try:
                    result = future.result()
                    results[registry_name] = result
                    logger.debug(
                        f"Completed {registry_name} discovery: {result.discovery_time:.2f}s"
                    )
                except Exception as e:
                    logger.error(f"Failed to discover {registry_name} registry: {e}")
                    results[registry_name] = RegistryDiscoveryResult(
                        registry_name=registry_name,
                        success=False,
                        discovery_time=0.0,
                        error=str(e),
                    )

        return results

    def _discover_sequential(
        self, use_cache: bool
    ) -> Dict[str, RegistryDiscoveryResult]:
        """Discover registries sequentially."""
        results = {}

        discovery_tasks = [
            ("model", self._discover_model_registry),
            ("dataset", self._discover_dataset_registry),
            ("metrics", self._discover_metrics_registry),
            ("processor", self._discover_processor_registry),
        ]

        logger.debug(
            f"Starting sequential discovery of {len(discovery_tasks)} registries"
        )

        for registry_name, discover_func in discovery_tasks:
            try:
                result = discover_func(use_cache)
                results[registry_name] = result
                logger.debug(
                    f"Completed {registry_name} discovery: {result.discovery_time:.2f}s"
                )
            except Exception as e:
                logger.error(f"Failed to discover {registry_name} registry: {e}")
                results[registry_name] = RegistryDiscoveryResult(
                    registry_name=registry_name,
                    success=False,
                    discovery_time=0.0,
                    error=str(e),
                )

        return results

    def _discover_model_registry(self, use_cache: bool) -> RegistryDiscoveryResult:
        """Discover model registry."""
        start_time = time.time()

        try:
            from karma.registries.model_registry import model_registry
            from karma.registries.cache_manager import get_cache_manager

            # Check if already discovered
            if model_registry._discovered:
                return RegistryDiscoveryResult(
                    registry_name="model",
                    success=True,
                    discovery_time=0.0,
                    from_cache=True,
                )

            # Check cache first
            from_cache = False
            if use_cache:
                cache_manager = get_cache_manager()
                cached_data = cache_manager.get_cached_discovery("model")
                from_cache = cached_data is not None

            # Perform discovery
            model_registry.discover_models(use_cache=use_cache)

            discovery_time = time.time() - start_time
            return RegistryDiscoveryResult(
                registry_name="model",
                success=True,
                discovery_time=discovery_time,
                from_cache=from_cache,
            )

        except Exception as e:
            discovery_time = time.time() - start_time
            return RegistryDiscoveryResult(
                registry_name="model",
                success=False,
                discovery_time=discovery_time,
                error=str(e),
            )

    def _discover_dataset_registry(self, use_cache: bool) -> RegistryDiscoveryResult:
        """Discover dataset registry."""
        start_time = time.time()

        try:
            from karma.registries.dataset_registry import dataset_registry
            from karma.registries.cache_manager import get_cache_manager

            # Check if already discovered
            if dataset_registry._discovered:
                return RegistryDiscoveryResult(
                    registry_name="dataset",
                    success=True,
                    discovery_time=0.0,
                    from_cache=True,
                )

            # Check cache first
            from_cache = False
            if use_cache:
                cache_manager = get_cache_manager()
                cached_data = cache_manager.get_cached_discovery("dataset")
                from_cache = cached_data is not None

            # Perform discovery
            dataset_registry.discover_datasets(use_cache=use_cache)

            discovery_time = time.time() - start_time
            return RegistryDiscoveryResult(
                registry_name="dataset",
                success=True,
                discovery_time=discovery_time,
                from_cache=from_cache,
            )

        except Exception as e:
            discovery_time = time.time() - start_time
            return RegistryDiscoveryResult(
                registry_name="dataset",
                success=False,
                discovery_time=discovery_time,
                error=str(e),
            )

    def _discover_metrics_registry(self, use_cache: bool) -> RegistryDiscoveryResult:
        """Discover metrics registry."""
        start_time = time.time()

        try:
            from karma.registries.metrics_registry import metric_registry
            from karma.registries.cache_manager import get_cache_manager

            # Check if already discovered
            if metric_registry._discovered:
                return RegistryDiscoveryResult(
                    registry_name="metrics",
                    success=True,
                    discovery_time=0.0,
                    from_cache=True,
                )

            # Check cache first
            from_cache = False
            if use_cache:
                cache_manager = get_cache_manager()
                cached_data = cache_manager.get_cached_discovery("metrics")
                from_cache = cached_data is not None

            # Perform discovery
            metric_registry.discover_metrics(use_cache=use_cache)

            discovery_time = time.time() - start_time
            return RegistryDiscoveryResult(
                registry_name="metrics",
                success=True,
                discovery_time=discovery_time,
                from_cache=from_cache,
            )

        except Exception as e:
            discovery_time = time.time() - start_time
            return RegistryDiscoveryResult(
                registry_name="metrics",
                success=False,
                discovery_time=discovery_time,
                error=str(e),
            )

    def _discover_processor_registry(self, use_cache: bool) -> RegistryDiscoveryResult:
        """Discover processor registry."""
        start_time = time.time()

        try:
            from karma.registries.processor_registry import processor_registry
            from karma.registries.cache_manager import get_cache_manager

            # Check if already discovered
            if processor_registry._discovered:
                return RegistryDiscoveryResult(
                    registry_name="processor",
                    success=True,
                    discovery_time=0.0,
                    from_cache=True,
                )

            # Check cache first
            from_cache = False
            if use_cache:
                cache_manager = get_cache_manager()
                cached_data = cache_manager.get_cached_discovery("processor")
                from_cache = cached_data is not None

            # Perform discovery
            processor_registry.discover_processors(use_cache=use_cache)

            discovery_time = time.time() - start_time
            return RegistryDiscoveryResult(
                registry_name="processor",
                success=True,
                discovery_time=discovery_time,
                from_cache=from_cache,
            )

        except Exception as e:
            discovery_time = time.time() - start_time
            return RegistryDiscoveryResult(
                registry_name="processor",
                success=False,
                discovery_time=discovery_time,
                error=str(e),
            )

    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get statistics about the last discovery operation."""
        return {
            "total_discovery_time": self._total_discovery_time,
            "registry_count": len(self._discovery_results),
            "successful_discoveries": sum(
                1 for r in self._discovery_results.values() if r.success
            ),
            "cache_hits": sum(
                1 for r in self._discovery_results.values() if r.from_cache
            ),
            "individual_results": {
                name: {
                    "success": result.success,
                    "discovery_time": result.discovery_time,
                    "from_cache": result.from_cache,
                    "error": result.error,
                }
                for name, result in self._discovery_results.items()
            },
        }


# Global registry manager instance
_global_registry_manager = None
_manager_lock = threading.RLock()


def get_registry_manager() -> RegistryManager:
    """Get the global registry manager instance."""
    global _global_registry_manager

    if _global_registry_manager is None:
        with _manager_lock:
            if _global_registry_manager is None:
                _global_registry_manager = RegistryManager()

    return _global_registry_manager


def discover_all_registries(
    use_cache: bool = True, parallel: bool = True
) -> Dict[str, RegistryDiscoveryResult]:
    """Convenience function to discover all registries."""
    manager = get_registry_manager()
    return manager.discover_all_registries(use_cache=False, parallel=False)


def get_discovery_stats() -> Dict[str, Any]:
    """Convenience function to get discovery statistics."""
    manager = get_registry_manager()
    return manager.get_discovery_stats()
