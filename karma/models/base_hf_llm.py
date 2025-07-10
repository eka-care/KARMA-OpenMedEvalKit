import logging
import os
from typing import Optional, List, Dict, Union

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.models.base_model_abs import BaseHFModel
from karma.data_models.model_meta import ModelMeta, ModalityType, ModelType
from karma.registries.model_registry import register_model_meta

logger = logging.getLogger(__name__)


class MedGemmaLLM(BaseHFModel):
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
            enable_thinking=False,  # MedGemma doesn't support thinking mode
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
            login(os.getenv("HUGGINGFACE_TOKEN"))
        except ValueError:
            logger.warning("HF token not found, will not login.")

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name_or_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )

    def run(self, inputs, **kwargs):
        model_inputs = self.preprocess(inputs)
        results = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        # Extract only the newly generated tokens
        input_length = model_inputs["input_ids"].shape[1]
        outputs = [
            self.processor.decode(results[i][input_length:], skip_special_tokens=True)
            for i in range(len(results))
        ]
        return self.postprocess(outputs)

    def preprocess(
        self,
        inputs: List[DataLoaderIterable],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch_messages = []

        for i, data_point in enumerate(inputs):
            messages = []
            user_content: List[Dict[str, Union[str, Image.Image]]] = [
                {"type": "text", "text": data_point.input}
            ]

            # Add image if provided
            if data_point.images:
                for image in data_point.images:
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

    def postprocess(self, outputs: List[str], **kwargs) -> List[str]:
        return [output.strip() for output in outputs]


MedGemmaModel = ModelMeta(
    name="google/medgemma-4b-it",
    description="Medgemma model",
    loader_class="karma.models.medgemma.MedGemmaLLM",
    loader_kwargs={
        "device": "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu",
        "max_tokens": 256,  # Sufficient for translation outputs
        "temperature": 0.01,  # Lower temperature for more consistent translations
        "top_p": 0.9,  # Nucleus sampling
        "top_k": 50,
    },
    revision=None,
    reference=None,
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    framework=["PyTorch", "Transformers"],
)
register_model_meta(MedGemmaModel)
