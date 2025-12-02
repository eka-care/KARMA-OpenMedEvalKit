import logging
from io import BytesIO
from typing import Optional, List, Dict, Union

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.models.base_model_abs import BaseModel
from karma.data_models.model_meta import ModelMeta, ModalityType, ModelType
from karma.registries.model_registry import register_model_meta

logger = logging.getLogger(__name__)


class MedGemmaLLM(BaseModel):
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
        super().__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=False,
            **kwargs,
        )

        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    @staticmethod
    def decode_image(image: bytes) -> Image.Image:
        return Image.open(BytesIO(image))

    def load_model(self):
        """Load HF model + ensure login."""
        from karma.utils.auth import ensure_hf_login

        ensure_hf_login()

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name_or_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )

    def run(self, inputs: List[DataLoaderIterable], **kwargs) -> List[str]:
        model_inputs = self.preprocess(inputs)

        results = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        input_len = model_inputs["input_ids"].shape[1]
        outputs = [
            self.processor.decode(results[i][input_len:], skip_special_tokens=True)
            for i in range(len(results))
        ]

        return self.postprocess(outputs)

    def preprocess(self, inputs: List[DataLoaderIterable], **kwargs) -> Dict[str, torch.Tensor]:
        batch_messages = []

        for data_point in inputs:
            messages = []

            # Conversation handling
            if data_point.conversation and len(data_point.conversation.conversation_turns) > 0:
                for turn in data_point.conversation.conversation_turns:
                    content = [{"type": "text", "text": turn.content}]
                    messages.append({"role": turn.role, "content": content})

                # Add images to last user message
                if data_point.images:
                    for msg in reversed(messages):
                        if msg["role"] == "user":
                            imgs = data_point.images if isinstance(data_point.images, list) else [data_point.images]
                            for img in imgs:
                                msg["content"].append({"type": "image", "image": self.decode_image(img)})
                            break

            elif data_point.input:
                # Simple text+image input
                user_content = [{"type": "text", "text": data_point.input}]

                if data_point.images:
                    imgs = data_point.images if isinstance(data_point.images, list) else [data_point.images]
                    for img in imgs:
                        user_content.append({"type": "image", "image": self.decode_image(img)})

                messages.append({"role": "user", "content": user_content})

            # Add system prompt
            if getattr(data_point, "system_prompt", None):
                messages.insert(
                    0, {"role": "system", "content": [{"type": "text", "text": data_point.system_prompt}]}
                )

            if not messages:
                logger.warning("No input or conversation for item — skipping")
                continue

            batch_messages.append(messages)

        processed = self.processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        return processed.to(self.device)

    def postprocess(self, outputs: List[str], **kwargs) -> List[str]:
        return [o.strip() for o in outputs]


# -------------------------
# Model Registration
# -------------------------

MedGemmaModel = ModelMeta(
    name="google/medgemma-4b-it",
    description="MedGemma model",
    loader_class="karma.models.medgemma.MedGemmaLLM",
    loader_kwargs={
        "device": "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
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
)

register_model_meta(MedGemmaModel)