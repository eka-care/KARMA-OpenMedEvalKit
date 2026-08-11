import logging
import os
import tempfile
from typing import Optional, List, Dict, Union

import torch
from PIL import Image
from io import BytesIO
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.models.base_model_abs import BaseModel
from karma.data_models.model_meta import ModelMeta, ModalityType, ModelType
from karma.registries.model_registry import register_model_meta

logger = logging.getLogger(__name__)


class Gemma4LLM(BaseModel):
    """Gemma 4 E-series multimodal model supporting text, image, and audio inputs."""

    def __init__(
        self,
        model_name_or_path,
        device: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        few_shot: bool = False,
        do_sample: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs,
        )
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.few_shot = few_shot
        self.do_sample = do_sample
        self._temp_files: List[str] = []

    @staticmethod
    def decode_image(image: bytes) -> Image.Image:
        return Image.open(BytesIO(image))

    def _audio_bytes_to_temp_path(self, audio: bytes) -> str:
        """Write audio bytes to a named temp file and return its path."""
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(audio)
        tmp.flush()
        tmp.close()
        self._temp_files.append(tmp.name)
        return tmp.name

    def _cleanup_temp_files(self):
        for path in self._temp_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._temp_files.clear()

    def load_model(self):
        from huggingface_hub import login

        # Imported lazily: AutoModelForMultimodalLM only exists in newer
        # transformers (the GPU training/serving env). Keeping it out of module
        # scope lets the meta register everywhere — e.g. so --cached-only can
        # re-judge cached outputs on a box without that transformers build.
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        try:
            login(os.getenv("HF_TOKEN"))
        except ValueError:
            logger.warning("HF token not found, will not login.")

        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_name_or_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )

    def run(self, inputs: List[DataLoaderIterable], **kwargs) -> List[str]:
        try:
            model_inputs = self.preprocess(inputs)
            results = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=self.do_sample,
            )

            input_length = model_inputs["input_ids"].shape[1]
            outputs = [
                self.processor.decode(results[i][input_length:], skip_special_tokens=True)
                for i in range(len(results))
            ]
            return self.postprocess(outputs)
        finally:
            self._cleanup_temp_files()

    def preprocess(
        self,
        inputs: List[DataLoaderIterable],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch_messages = []

        if self.few_shot:
            return self.processor.tokenizer(
                [inp.input for inp in inputs],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)

        for data_point in inputs:
            messages = []

            if data_point.conversation and len(data_point.conversation.conversation_turns) > 0:
                for turn in data_point.conversation.conversation_turns:
                    content = [{"type": "text", "text": turn.content}]
                    messages.append({"role": turn.role, "content": content})

                # Attach images/audio to the last user message
                if messages:
                    last_user = next(
                        (msg for msg in reversed(messages) if msg["role"] == "user"), None
                    )
                    if last_user is not None:
                        self._attach_media(last_user["content"], data_point)

            elif data_point.input:
                user_content: List[Dict] = []
                # Media content should precede text for optimal performance
                self._attach_media(user_content, data_point)
                user_content.append({"type": "text", "text": data_point.input})
                messages.append({"role": "user", "content": user_content})

            if hasattr(data_point, "system_prompt") and data_point.system_prompt:
                if not messages or messages[0]["role"] != "system":
                    messages.insert(
                        0,
                        {"role": "system", "content": [{"type": "text", "text": data_point.system_prompt}]},
                    )

            if not messages:
                logger.warning("No input or conversation data found for item, skipping")
                continue

            batch_messages.append(messages)

        model_inputs = self.processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device, dtype=torch.bfloat16)

        return model_inputs

    def _attach_media(self, content: List[Dict], data_point: DataLoaderIterable) -> None:
        """Append image and/or audio entries to a content list (in-place)."""
        if data_point.audio:
            audio_path = self._audio_bytes_to_temp_path(data_point.audio)
            content.append({"type": "audio", "url": audio_path})

        if data_point.images:
            images = (
                data_point.images
                if isinstance(data_point.images, list)
                else [data_point.images]
            )
            for image in images:
                content.append({"type": "image", "image": Gemma4LLM.decode_image(image)})

    def postprocess(self, outputs: List[str], **kwargs) -> List[str]:
        return [output.strip() for output in outputs]


_device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)

_default_kwargs = {
    "device": _device,
    "max_new_tokens": 2048,
    "temperature": 0.01,
    "top_p": 0.9,
    "top_k": 50,
    "few_shot": False,
    "do_sample": True,
}

Gemma4_E2B = ModelMeta(
    name="google/gemma-4-E2B-it",
    description="Gemma 4 E2B instruction-tuned multimodal model (text, image, audio)",
    loader_class="karma.models.gemma4.Gemma4LLM",
    loader_kwargs=_default_kwargs,
    revision=None,
    reference="https://huggingface.co/google/gemma-4-E2B-it",
    model_type=ModelType.MULTIMODAL,
    modalities=[ModalityType.TEXT, ModalityType.IMAGE, ModalityType.AUDIO],
)

Gemma4_E4B = ModelMeta(
    name="google/gemma-4-E4B-it",
    description="Gemma 4 E4B instruction-tuned multimodal model (text, image, audio)",
    loader_class="karma.models.gemma4.Gemma4LLM",
    loader_kwargs=_default_kwargs,
    revision=None,
    reference="https://huggingface.co/google/gemma-4-E4B-it",
    model_type=ModelType.MULTIMODAL,
    modalities=[ModalityType.TEXT, ModalityType.IMAGE, ModalityType.AUDIO],
)

register_model_meta(Gemma4_E2B)
register_model_meta(Gemma4_E4B)