"""
Model registry for automatic model discovery and registration.

This module provides a decorator-based registry system that allows models
to register themselves automatically when imported. Enhanced to support
ModelMeta configurations for comprehensive model metadata management.
"""

import importlib
import pkgutil
from typing import Dict, List, Any
import logging
import time

from karma.models.base_model_abs import BaseModel
from karma.data_models.model_meta import ModelMeta, ModelType, ModalityType

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Enhanced model registry supporting both legacy model classes and ModelMeta configurations.

    Provides automatic model discovery with comprehensive metadata management
    for multi-modal medical evaluation frameworks.
    """

    def __init__(self):
        self.model_metas: Dict[str, ModelMeta] = {}
        self.models_by_type: Dict[ModelType, List[str]] = {
            model_type: [] for model_type in ModelType
        }
        self.models_by_modality: Dict[ModalityType, List[str]] = {
            modality: [] for modality in ModalityType
        }
        self._discovered = False

    def register_model_meta(self, model_meta: ModelMeta) -> None:
        """
        Register a model using ModelMeta configuration.

        Args:
            model_meta: ModelMeta instance containing comprehensive model metadata

        Example:
            model_meta = ModelMeta(
                name="qwen-7b",
                model_type=ModelType.TEXT_GENERATION,
                loader_class="karma.models.qwen.QwenModel"
            )
            model_registry.register_model_meta(model_meta)
        """
        name = model_meta.name

        if name in self.model_metas:
            logger.warning(f"ModelMeta '{name}' is already registered. Overriding.")

        self.model_metas[name] = model_meta

        # Update categorical indexes
        if name not in self.models_by_type[model_meta.model_type]:
            self.models_by_type[model_meta.model_type].append(name)

        for modality in model_meta.modalities:
            if name not in self.models_by_modality[modality]:
                self.models_by_modality[modality].append(name)

        logger.debug(f"Registered ModelMeta: {name} -> {model_meta.model_type}")

    def get_model(self, name: str, **override_kwargs) -> BaseModel:
        """
        Get and instantiate model by name with optional parameter overrides.

        Args:
            name: Name of the model to retrieve
            **override_kwargs: Parameters to override model defaults

        Returns:
            Instantiated model instance

        Raises:
            ValueError: If model is not found
        """
        if not self._discovered:
            self.discover_models()

        if name in self.model_metas:
            m = self._get_model_from_meta(name, **override_kwargs)
            m.load_model()
            return m

        available = list(self.model_metas.keys())
        raise ValueError(f"Model '{name}' not found. Available models: {available}")

    def get_model_meta(self, name: str) -> ModelMeta:
        """
        Get ModelMeta configuration by name.

        Args:
            name: Name of the model metadata to retrieve

        Returns:
            ModelMeta configuration

        Raises:
            ValueError: If model metadata is not found
        """
        if not self._discovered:
            self.discover_models()

        if name not in self.model_metas:
            available = list(self.model_metas.keys())
            raise ValueError(f"ModelMeta '{name}' not found. Available: {available}")
        return self.model_metas[name]

    def _get_model_from_meta(self, name: str, **override_kwargs) -> BaseModel:
        """
        Load model using ModelMeta configuration with parameter overrides.

        Args:
            name: Model name
            **override_kwargs: Parameters to override defaults

        Returns:
            Instantiated model instance
        """
        model_meta = self.model_metas[name]
        model_class = model_meta.get_loader_class()

        # Merge kwargs: defaults < model_meta < overrides
        final_kwargs = model_meta.merge_kwargs(override_kwargs)

        # Always include model name/path
        final_kwargs["model_name_or_path"] = (
            model_meta.name if model_meta.model_path is None else model_meta.model_path
        )
        logger.info(f"Loading model '{name}' from {final_kwargs['model_name_or_path']}")
        return model_class(**final_kwargs)

    def list_models(self) -> List[str]:
        """
        List all available model names.

        Returns:
            List of all registered model names
        """
        if not self._discovered:
            self.discover_models()

        return sorted(list(self.model_metas.keys()))

    def list_models_by_type(self, model_type: ModelType) -> List[str]:
        """
        List models by type.

        Args:
            model_type: Model type to filter by

        Returns:
            List of model names of the specified type
        """
        if not self._discovered:
            self.discover_models()
        return self.models_by_type[model_type].copy()

    def list_models_by_modality(self, modality: ModalityType) -> List[str]:
        """
        List models by supported modality.

        Args:
            modality: Modality to filter by

        Returns:
            List of model names supporting the specified modality
        """
        if not self._discovered:
            self.discover_models()
        return self.models_by_modality[modality].copy()

    def get_models_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered models.

        Returns:
            List of dictionaries containing model information
        """
        if not self._discovered:
            self.discover_models()

        models_info = []

        for name, model_meta in self.model_metas.items():
            models_info.append(model_meta.get_model_info())

        return models_info

    def discover_models(self, use_cache: bool = True):
        """
        Automatically discover and import all model modules.

        This method imports all modules in the karma.models package,
        which triggers the decorator registration. Uses caching for performance.

        Args:
            use_cache: Whether to use cached discovery results (default: True)
        """
        if self._discovered:
            return

        # Try to load from cache first
        if use_cache:
            cached_data = self._load_from_cache()
            if cached_data:
                logger.debug("Loaded model registry from cache")
                self._discovered = True
                return

        # Perform discovery
        start_time = time.time()
        logger.debug("Starting model discovery...")

        try:
            import karma.models

            # Import all modules in karma.models package
            for finder, name, ispkg in pkgutil.iter_modules(
                karma.models.__path__, karma.models.__name__ + "."
            ):
                if not name.endswith(".base_model_abs") and not name.endswith(
                    ".model_meta"
                ):  # Skip base module to avoid issues
                    try:
                        importlib.import_module(name)
                        logger.info(f"Imported model module: {name}")
                    except ImportError as e:
                        logger.warning(f"Could not import model module {name}: {e}")
        except ImportError as e:
            logger.error(f"Could not import karma.models package: {e}")

        discovery_time = time.time() - start_time
        logger.debug(f"Model discovery completed in {discovery_time:.2f}s")

        self._discovered = True

        # Cache the results
        if use_cache:
            self._save_to_cache()

    def _load_from_cache(self) -> bool:
        """
        Load registry data from cache.

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        try:
            from karma.registries.cache_manager import get_cache_manager

            cache_manager = get_cache_manager()
            cached_data = cache_manager.get_cached_discovery("model")

            if cached_data:
                self.model_metas = cached_data.get("model_metas", {})
                self.models_by_type = cached_data.get(
                    "models_by_type", {model_type: [] for model_type in ModelType}
                )
                self.models_by_modality = cached_data.get(
                    "models_by_modality", {modality: [] for modality in ModalityType}
                )
                return True

        except Exception as e:
            logger.debug(f"Failed to load model registry from cache: {e}")

        return False

    def _save_to_cache(self) -> None:
        """
        Save current registry data to cache.
        """
        try:
            from karma.registries.cache_manager import get_cache_manager

            cache_manager = get_cache_manager()
            cache_data = {
                "model_metas": self.model_metas,
                "models_by_type": self.models_by_type,
                "models_by_modality": self.models_by_modality,
            }

            cache_manager.set_cached_discovery("model", cache_data)
            logger.debug("Saved model registry to cache")

        except Exception as e:
            logger.debug(f"Failed to save model registry to cache: {e}")

    def is_registered(self, name: str) -> bool:
        """
        Check if a model is registered.

        Args:
            name: Name of the model to check

        Returns:
            True if model is registered, False otherwise
        """
        if not self._discovered:
            self.discover_models()
        return name in self.model_metas

    def has_model_meta(self, name: str) -> bool:
        """
        Check if a model has ModelMeta configuration.

        Args:
            name: Name of the model to check

        Returns:
            True if model has metadata configuration, False otherwise
        """
        if not self._discovered:
            self.discover_models()
        return name in self.model_metas

    def unregister_model(self, name: str) -> bool:
        """
        Unregister a model.

        Args:
            name: Name of the model to unregister

        Returns:
            True if model was unregistered, False if it wasn't registered
        """
        if name in self.model_metas:
            model_meta = self.model_metas[name]
            del self.model_metas[name]

            # Remove from categorical indexes
            if name in self.models_by_type[model_meta.model_type]:
                self.models_by_type[model_meta.model_type].remove(name)

            for modality in model_meta.modalities:
                if name in self.models_by_modality[modality]:
                    self.models_by_modality[modality].remove(name)

            logger.debug(f"Unregistered model: {name}")
            return True

        return False

    def clear_registry(self):
        """Clear all registered models and metadata. Mainly for testing purposes."""
        self.model_metas.clear()

        # Reset categorical indexes
        for model_type in ModelType:
            self.models_by_type[model_type].clear()

        for modality in ModalityType:
            self.models_by_modality[modality].clear()

        self._discovered = False
        logger.debug("Cleared model registry")


# Global registry instance
model_registry = ModelRegistry()

# Convenience function
register_model_meta = model_registry.register_model_meta


# Convenience functions for CLI and programmatic access
def get_model(name: str, **kwargs) -> Any:
    """Get model instance with optional parameter overrides."""
    return model_registry.get_model(name, **kwargs)


def get_model_meta(name: str) -> ModelMeta:
    """Get ModelMeta configuration by name."""
    return model_registry.get_model_meta(name)


def list_models_by_type(model_type: ModelType) -> List[str]:
    """List models by type."""
    return model_registry.list_models_by_type(model_type)


def list_models_by_modality(modality: ModalityType) -> List[str]:
    """List models by modality."""
    return model_registry.list_models_by_modality(modality)
