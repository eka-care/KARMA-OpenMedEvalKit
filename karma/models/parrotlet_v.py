import logging
import os
from typing import Optional, List, Dict, Union

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from io import BytesIO
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.models.medgemma import MedGemmaLLM
from karma.data_models.model_meta import ModelMeta, ModalityType, ModelType
from karma.registries.model_registry import register_model_meta

logger = logging.getLogger(__name__)


class ParrotletVLiteLLM(MedGemmaLLM):
    """MedGemma language model with vision capabilities for medical applications."""

    def __init__(
        self,
        model_name_or_path,
        device: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize MedGemma LLM model.

        Args:
            model_name_or_path: Path to the model (HuggingFace model ID)
            device: Device to use for inference ("auto", "cuda", "cpu")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional model-specific parameters
        """
        # Initialize parent class
        super().__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs,
        )
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def load_model(self):
        # authenticate HF
        from huggingface_hub import login

        try:
            login(os.getenv("HF_TOKEN"))
        except ValueError:
            logger.warning("HF token not found, will not login.")

        self.model = AutoModel.from_pretrained(
            self.model_name_or_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )


ParrotletVLiteModel = ModelMeta(
    name="ekacare/parrotlet-v-lite-4b",
    description="Parrotlet-v-lite-4b model",
    loader_class="karma.models.parrotlet_v.ParrotletVLiteLLM",
    loader_kwargs={
        "device": "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu",
        "max_tokens": 1024,
        "temperature": 0.01,
        "top_p": 0.9, 
        "top_k": 50,
    },
    revision=None,
    reference=None,
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT, ModalityType.IMAGE],
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    framework=["PyTorch", "Transformers"],
)

register_model_meta(ParrotletVLiteModel)