import logging
from typing import Tuple, List, Optional, Dict
import torch

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.models.base_model_abs import BaseModel
from karma.data_models.model_meta import ModelMeta, ModelType, ModalityType
from karma.registries.model_registry import register_model_meta
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class QwenThinkingLLM(BaseModel):
    """Qwen language model with specialized thinking capabilities."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "mps",
        max_tokens: int = 32768,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        enable_thinking: bool = False,
        **kwargs,
    ):
        """
        Initialize Qwen Thinking LLM model.

        Args:
            model_path: Path to the model (HuggingFace model ID)
            device: Device to use for inference ("auto", "cuda", "cpu")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            enable_thinking: Whether to enable thinking capabilities
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
            enable_thinking=enable_thinking,
            **kwargs,
        )

        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.enable_thinking = enable_thinking
        # Qwen thinking end token ID (</think>)
        self.thinking_end_token_id = 151668

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
            if self.device == "cuda"
            else "eager",
        )
        self.processor = AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )

    def preprocess(
        self,
        inputs: List[DataLoaderIterable],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        processed_inputs = []

        for datapoint in inputs:
            messages = []

            # Check if conversation field exists and has data
            if (
                datapoint.conversation
                and len(datapoint.conversation.conversation_turns) > 0
            ):
                for turn in datapoint.conversation.conversation_turns:
                    # Map conversation turn to chat template format
                    messages.append({"role": turn.role, "content": turn.content})

            # Fall back to input field if no conversation data
            elif datapoint.input:
                messages = [{"role": "user", "content": datapoint.input}]

            # Add system prompt if available (for backward compatibility)
            if hasattr(datapoint, "system_prompt") and datapoint.system_prompt:
                # Insert system message at the beginning if not already present
                if not messages or messages[0]["role"] != "system":
                    messages.insert(
                        0, {"role": "system", "content": datapoint.system_prompt}
                    )

            # Ensure we have at least one message
            if not messages:
                logger.warning("No input or conversation data found for item, skipping")
                continue

            # Apply chat template
            formatted_input = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            processed_inputs.append(formatted_input)

        model_inputs = self.processor(
            processed_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
        ).to(self.device)
        return model_inputs

    def run(
        self,
        inputs: List[DataLoaderIterable],
        **kwargs,
    ) -> List[str]:
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

    def postprocess(self, model_outputs: List[str], **kwargs) -> List[str]:
        if not self.enable_thinking:
            return [output.strip() if output else "" for output in model_outputs]
        else:
            processed_outputs = []
            for output in model_outputs:
                if "</think>" in output:
                    # Extract only the final answer part, ignore thinking content
                    _, final_answer = self.parse_thinking_content(output)
                    processed_outputs.append(
                        final_answer.strip() if final_answer else ""
                    )
                else:
                    processed_outputs.append(output.strip() if output else "")
            return processed_outputs

    def parse_thinking_content(self, output) -> Tuple[str, str]:
        thinking_content = output.split("</think>")[0]
        final_answer = output.split("</think>")[1]
        return thinking_content.replace("<think>", "").strip(), final_answer.strip()


QwenModel3_06B = ModelMeta(
    name="Qwen/Qwen3-0.6B",
    description="QWEN model",
    loader_class="karma.models.qwen.QwenThinkingLLM",
    loader_kwargs={
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "enable_thinking": True,
        "max_tokens": 32768,
    },
    revision=None,
    reference=None,
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
)
QwenModel_1_7B = ModelMeta(
    name="Qwen/Qwen3-1.7B",
    description="QWEN model",
    loader_class="karma.models.qwen.QwenThinkingLLM",
    loader_kwargs={
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "enable_thinking": True,
        "max_tokens": 32768,
    },
    revision=None,
    reference=None,
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
)
register_model_meta(QwenModel3_06B)
register_model_meta(QwenModel_1_7B)
