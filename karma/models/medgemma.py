import logging
from typing import Optional, List, Dict, Union

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from IPython.display import Audio

from karma.models.base import BaseLLM
from karma.registries.model_registry import register_model

logger = logging.getLogger(__name__)


@register_model("medgemma")
class MedGemmaLLM(BaseLLM):
    """MedGemma language model with vision capabilities for medical applications."""

    def __init__(
        self,
        model_path: str = "google/medgemma-4b-it",
        device: str = "auto",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize MedGemma LLM model.

        Args:
            model_path: Path to the model (HuggingFace model ID)
            device: Device to use for inference ("auto", "cuda", "cpu")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional model-specific parameters
        """
        # Initialize parent class
        super().__init__(
            model_path=model_path,
            device=device,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=False,  # MedGemma doesn't support thinking mode
            **kwargs,
        )

    def load_model(self):
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    def format_inputs(
        self,
        prompts: List[str],
        images: Optional[List[List[Image.Image]]] = None,
        audios: Optional[List[Audio]] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_messages = []

        for i, prompt in enumerate(prompts):
            messages = []
            user_content: List[Dict[str, Union[str, Image.Image]]] = [
                {"type": "text", "text": prompt}
            ]

            # Add image if provided
            if images is not None and i < len(images) and images[i] is not None:
                sample_images = images[i]
                for image in sample_images:
                    user_content.append({"type": "image", "image": image})

            messages.append({"role": "user", "content": user_content})

            batch_messages.append(messages)

        # Process all messages in batch
        inputs = self.processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device, dtype=torch.bfloat16)

        return inputs

    def format_outputs(self, outputs: List[str]) -> List[str]:
        return [output.strip() for output in outputs]
