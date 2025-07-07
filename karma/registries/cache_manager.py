"""
Registry cache manager for optimizing discovery performance.

This module provides caching functionality for registry discovery operations
to significantly reduce CLI startup time by avoiding repeated module imports.
"""

import time
import os
import pickle
import threading
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RegistryCache:
    """
    Thread-safe cache manager for registry discovery results.
    
    Provides persistent caching with TTL (Time To Live) support and
    automatic cache invalidation based on file modification times.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, default_ttl: int = 300):
        """
        Initialize registry cache manager.
        
        Args:
            cache_dir: Directory to store cache files (default: ~/.karma/cache)
            default_ttl: Default TTL in seconds (default: 5 minutes)
        """
        self.default_ttl = default_ttl
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".karma" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for current session
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.RLock()
        
        logger.debug(f"Registry cache initialized with dir: {self.cache_dir}")
    
    def _get_cache_file(self, registry_name: str) -> Path:
        """Get cache file path for a registry."""
        return self.cache_dir / f"{registry_name}_cache.pkl"
    
    def _get_registry_paths(self, registry_name: str) -> List[Path]:
        """Get source paths to monitor for changes."""
        if registry_name == "model":
            return [Path(__file__).parent.parent / "models"]
        elif registry_name == "dataset":
            return [Path(__file__).parent.parent / "eval_datasets"]
        elif registry_name == "metrics":
            return [Path(__file__).parent.parent / "metrics"]
        elif registry_name == "processor":
            return [Path(__file__).parent.parent / "processors"]
        else:
            return []
    
    def _get_source_modification_time(self, registry_name: str) -> float:
        """Get the latest modification time of source files."""
        paths = self._get_registry_paths(registry_name)
        max_mtime = 0.0
        
        for path in paths:
            if path.exists():
                for file_path in path.rglob("*.py"):
                    try:
                        mtime = file_path.stat().st_mtime
                        max_mtime = max(max_mtime, mtime)
                    except OSError:
                        continue
        
        return max_mtime
    
    def _is_cache_valid(self, registry_name: str, cache_data: Dict[str, Any]) -> bool:
        """Check if cache is still valid based on TTL and source modification times."""
        # Check TTL
        cache_time = cache_data.get("timestamp", 0)
        ttl = cache_data.get("ttl", self.default_ttl)
        
        if time.time() - cache_time > ttl:
            logger.debug(f"Cache for {registry_name} expired (TTL: {ttl}s)")
            return False
        
        # Check source modification times
        cached_mtime = cache_data.get("source_mtime", 0)
        current_mtime = self._get_source_modification_time(registry_name)
        
        if current_mtime > cached_mtime:
            logger.debug(f"Cache for {registry_name} invalidated (source files changed)")
            return False
        
        return True
    
    def get_cached_discovery(self, registry_name: str) -> Optional[Dict[str, Any]]:
        """
        Get cached discovery results for a registry.
        
        Args:
            registry_name: Name of the registry (model, dataset, metrics, processor)
            
        Returns:
            Cached discovery data if valid, None otherwise
        """
        with self._cache_lock:
            # Check memory cache first
            if registry_name in self._memory_cache:
                cache_data = self._memory_cache[registry_name]
                if self._is_cache_valid(registry_name, cache_data):
                    logger.debug(f"Cache hit (memory) for {registry_name}")
                    return cache_data.get("data")
                else:
                    # Remove invalid cache
                    del self._memory_cache[registry_name]
            
            # Check disk cache
            cache_file = self._get_cache_file(registry_name)
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    if self._is_cache_valid(registry_name, cache_data):
                        # Load into memory cache
                        self._memory_cache[registry_name] = cache_data
                        logger.debug(f"Cache hit (disk) for {registry_name}")
                        return cache_data.get("data")
                    else:
                        # Remove invalid cache file
                        try:
                            cache_file.unlink()
                        except OSError:
                            pass
                
                except (pickle.PickleError, EOFError, OSError) as e:
                    logger.warning(f"Failed to load cache for {registry_name}: {e}")
                    try:
                        cache_file.unlink()
                    except OSError:
                        pass
            
            logger.debug(f"Cache miss for {registry_name}")
            return None
    
    def set_cached_discovery(self, registry_name: str, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """
        Cache discovery results for a registry.
        
        Args:
            registry_name: Name of the registry
            data: Discovery data to cache
            ttl: Time to live in seconds (default: use default_ttl)
        """
        if ttl is None:
            ttl = self.default_ttl
        
        cache_data = {
            "data": data,
            "timestamp": time.time(),
            "ttl": ttl,
            "source_mtime": self._get_source_modification_time(registry_name)
        }
        
        with self._cache_lock:
            # Store in memory cache
            self._memory_cache[registry_name] = cache_data
            
            # Store in disk cache
            cache_file = self._get_cache_file(registry_name)
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                logger.debug(f"Cached discovery results for {registry_name}")
            except (pickle.PickleError, OSError) as e:
                logger.warning(f"Failed to save cache for {registry_name}: {e}")
    
    def invalidate_cache(self, registry_name: str) -> None:
        """
        Invalidate cache for a specific registry.
        
        Args:
            registry_name: Name of the registry to invalidate
        """
        with self._cache_lock:
            # Remove from memory cache
            if registry_name in self._memory_cache:
                del self._memory_cache[registry_name]
            
            # Remove disk cache file
            cache_file = self._get_cache_file(registry_name)
            try:
                if cache_file.exists():
                    cache_file.unlink()
                    logger.debug(f"Invalidated cache for {registry_name}")
            except OSError as e:
                logger.warning(f"Failed to remove cache file for {registry_name}: {e}")
    
    def clear_all_caches(self) -> None:
        """Clear all cached discovery results."""
        with self._cache_lock:
            self._memory_cache.clear()
            
            # Remove all cache files
            for cache_file in self.cache_dir.glob("*_cache.pkl"):
                try:
                    cache_file.unlink()
                except OSError:
                    pass
            
            logger.debug("Cleared all registry caches")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cache usage."""
        with self._cache_lock:
            stats = {
                "memory_cache_size": len(self._memory_cache),
                "cache_dir": str(self.cache_dir),
                "disk_cache_files": len(list(self.cache_dir.glob("*_cache.pkl"))),
                "default_ttl": self.default_ttl
            }
            
            # Add individual cache info
            for registry_name in self._memory_cache:
                cache_data = self._memory_cache[registry_name]
                stats[f"{registry_name}_cache"] = {
                    "timestamp": cache_data.get("timestamp", 0),
                    "ttl": cache_data.get("ttl", self.default_ttl),
                    "age_seconds": time.time() - cache_data.get("timestamp", 0)
                }
            
            return stats


# Global cache instance
_global_cache = None
_cache_lock = threading.RLock()


def get_cache_manager() -> RegistryCache:
    """Get the global registry cache manager instance."""
    global _global_cache
    
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = RegistryCache()
    
    return _global_cache


def invalidate_registry_cache(registry_name: str) -> None:
    """Convenience function to invalidate a specific registry cache."""
    cache_manager = get_cache_manager()
    cache_manager.invalidate_cache(registry_name)


def clear_all_registry_caches() -> None:
    """Convenience function to clear all registry caches."""
    cache_manager = get_cache_manager()
    cache_manager.clear_all_caches()