from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from enum import Enum
import importlib


class ModelType(str, Enum):
    """Enumeration of supported model types."""

    TEXT_GENERATION = "text_generation"
    AUDIO_RECOGNITION = "audio_recognition"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"


class ModalityType(str, Enum):
    """Enumeration of supported input/output modalities."""

    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"


class ModelMeta(BaseModel):
    """
    Comprehensive model metadata configuration supporting multi-modal models.

    Inspired by MTEB's ModelMeta but extended for medical evaluation contexts
    and multi-modal capabilities including text generation, audio recognition,
    and embedding models.
    """

    # Core identification
    name: str = Field(..., description="HuggingFace-style model identifier")
    model_path: Optional[str] = Field(
        default=None, description="Path to the local model"
    )
    revision: Optional[str] = Field(
        default=None, description="Git commit hash or version tag"
    )
    reference: Optional[str] = Field(
        default=None, description="Official documentation or homepage URL"
    )
    description: Optional[str] = Field(
        default=None, description="Brief model description"
    )

    # Model loading configuration
    loader_class: str = Field(..., description="Full Python path to model loader class")
    loader_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Default arguments for model loader"
    )

    # Evaluation configuration
    default_eval_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Default evaluation parameters"
    )

    # Model type classification
    model_type: ModelType = Field(..., description="Primary model type classification")
    modalities: List[ModalityType] = Field(
        default_factory=lambda: [ModalityType.TEXT],
        description="Supported input/output modalities",
    )



    # Language and domain support
    languages: Optional[List[str]] = Field(
        default_factory=lambda: ["eng-Latn"],
        description="Supported languages in ISO format",
    )

    # Release information
    release_date: Optional[str] = Field(
        default=None, description="Model release date (YYYY-MM-DD)"
    )
    version: Optional[str] = Field(default=None, description="Model version")

    @validator("languages")
    def validate_languages(cls, v):
        """Validate language codes format."""
        if not v:
            return ["eng-Latn"]
        return v

    @validator("loader_class")
    def validate_loader_class(cls, v):
        """Validate that loader class path is importable."""
        try:
            module_path, class_name = v.rsplit(".", 1)
            module = importlib.import_module(module_path)
            getattr(module, class_name)
        except (ImportError, AttributeError, ValueError):
            raise ValueError(f"Invalid loader class path: {v}")
        return v

    def get_loader_class(self):
        """
        Import and return the actual loader class.

        Returns:
            The model loader class
        """
        module_path, class_name = self.loader_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def merge_kwargs(
        self, override_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Merge loader kwargs with override parameters.

        Args:
            override_kwargs: Parameters to override defaults

        Returns:
            Merged kwargs dictionary with override precedence
        """
        merged = self.loader_kwargs.copy()
        if override_kwargs:
            merged.update(override_kwargs)
        return merged

    def is_compatible_with_modality(self, modality: ModalityType) -> bool:
        """
        Check if model supports a specific modality.

        Args:
            modality: Modality to check

        Returns:
            True if modality is supported
        """
        return modality in self.modalities

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get formatted model information for display.

        Returns:
            Dictionary with key model information
        """
        return {
            "name": self.name,
            "type": self.model_type,
            "modalities": [m for m in self.modalities],
            "languages": self.languages,
            "version": self.version or self.revision,
        }

    class Config:
        use_enum_values = True
        validate_assignment = True
        extra = "allow"
