"""
Gemma / MedGemma HuggingFace-local inference model.

A single arch-dispatching class that auto-detects whether the checkpoint is
text-only or multimodal by inspecting ``AutoConfig.architectures[0]``:

* Architectures ending in ``ForCausalLM``  -> ``AutoModelForCausalLM`` +
  ``AutoTokenizer`` (text-only branch).
* Everything else (``ForConditionalGeneration``, ``ForImageTextToText``, ...)
  -> ``AutoModelForImageTextToText`` + ``AutoProcessor`` (multimodal branch).

Callers can force a specific branch via the ``text_only`` kwarg. This class
does NOT replace ``MedGemmaLLM`` (which remains the loader for
``google/medgemma-4b-it``); instead it registers 4 new Gemma / MedGemma
variants. See the module-scope ``ModelMeta`` registrations at the bottom of
this file for the exact IDs.

Design notes:
* ``transformers==4.53.0`` (the repo pin) supports ``Gemma3*`` but not
  ``Gemma4*`` — ``google/gemma-4-26B-A4B`` is deferred to a later scoped
  change that bumps the pin.
* Weight downloads require ``HF_TOKEN`` in the environment plus prior
  license acceptance in the HF web UI (``google/medgemma-*`` = auto-accept
  health-ai-developer-foundations; ``google/gemma-3-*`` = manual Gemma
  license). Missing token -> soft-fail with a WARNING log (mirrors
  ``MedGemmaLLM.load_model``).
"""

import logging
import os
from io import BytesIO
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.data_models.model_meta import ModalityType, ModelMeta, ModelType
from karma.models.base_model_abs import BaseModel
from karma.registries.model_registry import register_model_meta

logger = logging.getLogger(__name__)


class GemmaHFLLM(BaseModel):
    """Arch-dispatching HuggingFace-local loader for Gemma / MedGemma variants.

    Extends :class:`karma.models.base_model_abs.BaseModel`. Same kwarg shape
    as :class:`karma.models.medgemma.MedGemmaLLM` plus an explicit
    ``text_only`` override for the text-vs-multimodal dispatch.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.01,
        top_p: float = 0.9,
        top_k: int = 50,
        few_shot: bool = False,
        do_sample: bool = True,
        text_only: Optional[bool] = None,
        **kwargs,
    ):
        """Initialize the Gemma HF model wrapper.

        Args:
            model_name_or_path: HuggingFace model ID (e.g. ``google/gemma-3-4b-it``).
            device: Target device ("cuda", "mps", "cpu", or "auto").
            max_new_tokens: Generation budget per sample.
            temperature: Sampling temperature (0.01 ~= near-greedy).
            top_p: Nucleus sampling threshold.
            top_k: Top-k filter.
            few_shot: If True, skip chat-template formatting in preprocess
                and batch inputs as-is. Mirrors ``MedGemmaLLM.preprocess``.
            do_sample: Whether to sample at generation time.
            text_only: If ``None`` (default), auto-detect via
                ``AutoConfig.architectures[0]``. If ``True`` / ``False``,
                force the text-only / multimodal branch respectively.
            **kwargs: Passed through to :class:`BaseModel`.
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=False,  # Gemma family does not expose thinking mode.
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
        self.text_only = text_only

        # Populated by load_model().
        self._is_mm: Optional[bool] = None
        self.processor = None  # MM branch only.
        self.tokenizer = None  # Text branch only.

    @staticmethod
    def decode_image(image: bytes) -> Image.Image:
        """Decode raw image bytes into a PIL Image (used by the MM branch)."""
        return Image.open(BytesIO(image))

    def load_model(self):
        """Detect architecture, pick the right AutoModel* pair, and load.

        Flow:
            1. Soft-login to HF (``HF_TOKEN`` env var; warn if missing).
            2. ``AutoConfig.from_pretrained`` to read ``architectures``.
            3. Dispatch: ``*ForCausalLM`` -> text branch, else MM branch.
               An explicit ``self.text_only`` in ``__init__`` overrides.
            4. Load weights at ``bfloat16`` with the resolved ``device_map``.
            5. Record the branch on ``self._is_mm`` for preprocess/run.
        """
        # 1. HF auth. Soft-fail if token missing (matches MedGemmaLLM).
        from huggingface_hub import login

        try:
            login(os.getenv("HF_TOKEN"))
        except ValueError:
            logger.warning("HF token not found, will not login.")

        # 2. Config sniff -> architecture name.
        cfg = AutoConfig.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )
        arch = (cfg.architectures or [""])[0]

        # 3. Dispatch.
        if self.text_only is not None:
            is_text_only = bool(self.text_only)
            logger.info(
                "GemmaHFLLM: text_only forced to %s (arch=%s)", is_text_only, arch
            )
        else:
            is_text_only = arch.endswith("ForCausalLM")
            logger.info(
                "GemmaHFLLM: auto-detected is_text_only=%s from arch=%s",
                is_text_only,
                arch,
            )

        # 4. Load weights.
        if is_text_only:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
            # Left-padding is required for causal-LM batched generate(); Gemma
            # tokenizers ship without a pad_token, so fall back to eos_token.
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name_or_path,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )

        # 5. Record branch for preprocess/run dispatch.
        self._is_mm = not is_text_only
        self.is_loaded = True

    def preprocess(
        self,
        inputs: List[DataLoaderIterable],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Convert KARMA DataLoaderIterables into model-ready tensors.

        Branches on ``self._is_mm``:
            * MM: mirror ``MedGemmaLLM.preprocess`` — build per-turn content
              lists with text + optional images, then
              ``processor.apply_chat_template(...)``.
            * Text: build role/content dicts (strings only) and use
              ``tokenizer.apply_chat_template(...)``.
        """
        if self._is_mm:
            return self._preprocess_mm(inputs, **kwargs)
        return self._preprocess_text(inputs, **kwargs)

    def _preprocess_mm(
        self,
        inputs: List[DataLoaderIterable],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Multimodal preprocess — verbatim shape of MedGemmaLLM.preprocess."""
        batch_messages = []

        if self.few_shot:
            batch_messages = self.processor.tokenizer(
                [item.input for item in inputs],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)
            return batch_messages

        for data_point in inputs:
            messages = []

            if (
                data_point.conversation
                and len(data_point.conversation.conversation_turns) > 0
            ):
                for turn in data_point.conversation.conversation_turns:
                    content = [{"type": "text", "text": turn.content}]
                    messages.append({"role": turn.role, "content": content})

                if data_point.images and messages:
                    for msg in reversed(messages):
                        if msg["role"] == "user":
                            if isinstance(data_point.images, list):
                                for image in data_point.images:
                                    msg["content"].append(
                                        {
                                            "type": "image",
                                            "image": GemmaHFLLM.decode_image(image),
                                        }
                                    )
                            else:
                                msg["content"].append(
                                    {
                                        "type": "image",
                                        "image": GemmaHFLLM.decode_image(
                                            data_point.images
                                        ),
                                    }
                                )
                            break

            elif data_point.input:
                user_content: List[Dict[str, Union[str, Image.Image]]] = [
                    {"type": "text", "text": data_point.input}
                ]
                if data_point.images:
                    if isinstance(data_point.images, list):
                        for image in data_point.images:
                            user_content.append(
                                {
                                    "type": "image",
                                    "image": GemmaHFLLM.decode_image(image),
                                }
                            )
                    else:
                        user_content.append(
                            {
                                "type": "image",
                                "image": GemmaHFLLM.decode_image(data_point.images),
                            }
                        )
                messages.append({"role": "user", "content": user_content})

            if (
                hasattr(data_point, "system_prompt")
                and data_point.system_prompt
            ):
                if not messages or messages[0]["role"] != "system":
                    messages.insert(
                        0,
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": data_point.system_prompt}
                            ],
                        },
                    )

            if not messages:
                logger.warning(
                    "No input or conversation data found for item, skipping"
                )
                continue

            batch_messages.append(messages)

        processed = self.processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device, dtype=torch.bfloat16)

        return processed

    def _preprocess_text(
        self,
        inputs: List[DataLoaderIterable],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Text-only preprocess — string content, tokenizer chat template."""
        if self.few_shot:
            return self.tokenizer(
                [item.input for item in inputs],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)

        batch_messages = []
        for data_point in inputs:
            messages = []

            if (
                data_point.conversation
                and len(data_point.conversation.conversation_turns) > 0
            ):
                for turn in data_point.conversation.conversation_turns:
                    messages.append({"role": turn.role, "content": turn.content})
            elif data_point.input:
                messages.append({"role": "user", "content": data_point.input})

            if (
                hasattr(data_point, "system_prompt")
                and data_point.system_prompt
            ):
                if not messages or messages[0]["role"] != "system":
                    messages.insert(
                        0,
                        {"role": "system", "content": data_point.system_prompt},
                    )

            if not messages:
                logger.warning(
                    "No input or conversation data found for item, skipping"
                )
                continue

            batch_messages.append(messages)

        # apply_chat_template with batched inputs returns a single padded tensor.
        input_ids = self.tokenizer.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long() if (
            self.tokenizer.pad_token_id is not None
        ) else torch.ones_like(input_ids)

        return {
            "input_ids": input_ids.to(self.model.device),
            "attention_mask": attention_mask.to(self.model.device),
        }

    def run(self, inputs: List[DataLoaderIterable], **kwargs) -> List[str]:
        """Generate completions and return decoded strings.

        Dispatches decoding on ``self._is_mm`` — MM uses
        ``self.processor.decode``, text uses ``self.tokenizer.decode``.
        """
        model_inputs = self.preprocess(inputs)
        results = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            do_sample=self.do_sample,
        )

        if "input_ids" not in model_inputs:
            raise KeyError(
                "GemmaHFLLM.run: preprocess output missing 'input_ids' "
                f"(keys={list(model_inputs.keys())})"
            )
        input_length = model_inputs["input_ids"].shape[1]
        if self._is_mm:
            decoder = self.processor.decode
        else:
            decoder = self.tokenizer.decode

        outputs = [
            decoder(results[i][input_length:], skip_special_tokens=True)
            for i in range(len(results))
        ]
        return self.postprocess(outputs)

    def postprocess(self, outputs: List[str], **kwargs) -> List[str]:
        """Strip whitespace on decoded strings (matches ``MedGemmaLLM``)."""
        return [output.strip() for output in outputs]


# ---------------------------------------------------------------------------
# ModelMeta registrations (4 IDs).
# Shared default loader_kwargs: same device ternary + generation defaults as
# ``karma/models/medgemma.py:185-206``. The 27B text-only variant also sets
# ``text_only=True`` as belt-and-braces — auto-detect would do the right
# thing on its own, but the explicit flag documents the intent and guards
# against any future HF config reshuffle.
# ---------------------------------------------------------------------------

_SHARED_GEMMA_LOADER_KWARGS = {
    "device": "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu",
    "max_new_tokens": 1024,
    "temperature": 0.01,
    "top_p": 0.9,
    "top_k": 50,
    "few_shot": False,
    "do_sample": True,
}


MedGemma15_4B_IT = ModelMeta(
    name="google/medgemma-1.5-4b-it",
    description="MedGemma 1.5 4B instruction-tuned (text + vision).",
    loader_class="karma.models.gemma_hf.GemmaHFLLM",
    loader_kwargs=dict(_SHARED_GEMMA_LOADER_KWARGS),
    revision=None,
    reference=None,
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT, ModalityType.IMAGE],
)
register_model_meta(MedGemma15_4B_IT)


Gemma3_4B_IT = ModelMeta(
    name="google/gemma-3-4b-it",
    description="Gemma 3 4B instruction-tuned (text + vision).",
    loader_class="karma.models.gemma_hf.GemmaHFLLM",
    loader_kwargs=dict(_SHARED_GEMMA_LOADER_KWARGS),
    revision=None,
    reference=None,
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT, ModalityType.IMAGE],
)
register_model_meta(Gemma3_4B_IT)


MedGemma27B_TextIT = ModelMeta(
    name="google/medgemma-27b-text-it",
    description="MedGemma 27B text-only instruction-tuned.",
    loader_class="karma.models.gemma_hf.GemmaHFLLM",
    loader_kwargs={
        **_SHARED_GEMMA_LOADER_KWARGS,
        "text_only": True,  # Belt-and-braces; auto-detect agrees.
    },
    revision=None,
    reference=None,
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
)
register_model_meta(MedGemma27B_TextIT)


Gemma3_27B_IT = ModelMeta(
    name="google/gemma-3-27b-it",
    description="Gemma 3 27B instruction-tuned (text + vision).",
    loader_class="karma.models.gemma_hf.GemmaHFLLM",
    loader_kwargs=dict(_SHARED_GEMMA_LOADER_KWARGS),
    revision=None,
    reference=None,
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT, ModalityType.IMAGE],
)
register_model_meta(Gemma3_27B_IT)
