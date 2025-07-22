from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseModel(ABC):
    """
    Abstract base class for HuggingFace-based models supporting LLM, Audio, and Embedding models.

    This class provides a unified interface for different model types while maintaining
    flexibility for model-specific implementations.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the base model.

        Args:
            model_name_or_path: HuggingFace model identifier or local path
            model_type: Type of model (LLM, AUDIO, EMBEDDING)
            device: Device to run the model on ('cuda', 'cpu', etc.)
            **kwargs: Additional model-specific arguments
        """
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.is_loaded = False

    @abstractmethod
    def load_model(self, **kwargs) -> None:
        """
        Load the model and tokenizer/processor.

        Implementation should:
        - Load the appropriate model class for the model type
        - Load tokenizer/processor if needed
        - Move model to specified device
        - Set self.is_loaded = True

        Args:
            **kwargs: Model-specific loading arguments
        """
        pass

    @abstractmethod
    def run(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor], List[str], Any],
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the model.

        This is the core method that must be implemented by all model types.

        Args:
            inputs: Model inputs (format depends on model type)
                   - LLM: tokenized text or raw strings
                   - Audio: audio tensors or preprocessed features
                   - Embedding: text or tokens to embed
            **kwargs: Additional forward pass arguments

        Returns:
            Model outputs (format depends on model type)
            - LLM: logits, generated text, or model outputs
            - Audio: classification scores, transcriptions, etc.
            - Embedding: embedding vectors
        """
        pass

    @abstractmethod
    def preprocess(
        self,
        inputs: List[Any],
        **kwargs,
    ):
        """
        Preprocess raw inputs into model-ready format.

        Args:
            raw_inputs: Raw input data

        Returns:
            Preprocessed inputs ready for forward pass
            :param prompts:
        """
        pass

    @abstractmethod
    def postprocess(self, model_outputs: Any, **kwargs) -> Any:
        """
        Postprocess model outputs into final format.

        Args:
            model_outputs: Raw model outputs from forward pass
            **kwargs: Additional postprocessing arguments

        Returns:
            Processed outputs in desired format
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        return {
            "model_name_or_path": self.model_name_or_path,
            "model_type": self.model_type.value,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "num_parameters": self.get_num_parameters() if self.is_loaded else None,
        }

    def get_num_parameters(self) -> int:
        """
        Get the number of trainable parameters in the model.

        Returns:
            Number of trainable parameters
        """
        if not self.is_loaded or self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def to(self, device: str) -> "BaseModel":
        """
        Move model to specified device.

        Args:
            device: Target device

        Returns:
            Self for method chaining
        """
        if self.is_loaded and self.model is not None:
            self.model = self.model.to(device)
        self.device = device
        return self

    def eval(self) -> "BaseModel":
        """
        Set model to evaluation mode.

        Returns:
            Self for method chaining
        """
        if self.is_loaded and self.model is not None:
            self.model.eval()
        return self

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name_or_path='{self.model_name_or_path}', "
            f"device='{self.device}', "
        )
