"""
Base LLM class with utility functions for all model implementations.
"""

import logging
from typing import List, Optional, Any, Dict
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from dataclasses import dataclass, asdict
from abc import abstractmethod
from PIL import Image
from IPython.display import Audio


logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration for caching."""

    model_id: str
    temperature: float = 0.7
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    enable_thinking: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for hashing."""
        return asdict(self)


class BaseLLM:
    """Base class for all LLM implementations with common utility functions."""

    def __init__(
        self,
        model_path: str,
        device: str = "mps",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        enable_thinking: bool = False,
        **kwargs,
    ):
        """
        Initialize base LLM.

        Args:
            model_path: Path to the model (required)
            device: Device to use for inference ("auto", "cuda", "cpu")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            enable_thinking: Whether to enable thinking mode
            **kwargs: Additional model-specific parameters
        """
        self.model_path = model_path
        self.device = device
        self.kwargs = kwargs
        self.model_config = ModelConfig(
            model_id=model_path.split("/")[-1],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        self.load_model()

    def load_model(self):
        """Load the model. To be implemented by subclasses."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    @abstractmethod
    def format_inputs(
        self,
        prompts: List[str],
        images: Optional[List[List[Image.Image]]] = None,
        audios: Optional[List[Audio]] = None,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def format_outputs(self, outputs: List[torch.Tensor]) -> List[str]:
        pass

    def generate(
        self,
        prompt: str,
        image: Optional[List[Image.Image]] = None,
        audio: Optional[Audio] = None,
        **kwargs,
    ) -> str:
        """
        Generate text using the model (returns only the final output).
        For compatibility with DeepEval's base interface.

        To be implemented by subclasses.
        """
        model_inputs = self.format_inputs(
            [prompt],
            [image] if image is not None else None,
            [audio] if audio is not None else None,
        )
        results = self.model.generate(
            **model_inputs,
            max_new_tokens=self.model_config.max_tokens,
            temperature=self.model_config.temperature,
            top_p=self.model_config.top_p,
            top_k=self.model_config.top_k,
        )

        # Extract only the newly generated tokens
        input_length = model_inputs["input_ids"].shape[1]
        outputs = [
            self.processor.decode(results[i][input_length:], skip_special_tokens=True)
            for i in range(len(results))
        ]
        formatted_outputs = self.format_outputs(outputs)

        # Return only the final answer for single generation

        return formatted_outputs[0]

    def batch_generate(
        self,
        prompts: List[str],
        images: Optional[List[List[Image.Image]]] = None,
        audios: Optional[List[Audio]] = None,
        **kwargs,
    ) -> List[str]:
        model_inputs = self.format_inputs(prompts, images, audios)
        results = self.model.generate(
            **model_inputs,
            max_new_tokens=self.model_config.max_tokens,
            temperature=self.model_config.temperature,
            top_p=self.model_config.top_p,
            top_k=self.model_config.top_k,
        )

        # Extract only the newly generated tokens
        input_length = model_inputs["input_ids"].shape[1]
        outputs = [
            self.processor.decode(results[i][input_length:], skip_special_tokens=True)
            for i in range(len(results))
        ]
        return self.format_outputs(outputs)
