import logging
from typing import Tuple, List, Optional
import torch
from PIL import Image
from IPython.display import Audio

from karma.models.base import BaseLLM
from karma.registries.model_registry import register_model
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@register_model("qwen")
class QwenThinkingLLM(BaseLLM):
    """Qwen language model with specialized thinking capabilities."""

    def __init__(
        self,
        model_path: str,
        device: str = "mps",
        max_tokens: int = 32768,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        enable_thinking: bool = True,
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
            model_path=model_path,
            device=device,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
            **kwargs,
        )

        # Qwen thinking end token ID (</think>)
        self.thinking_end_token_id = 151668

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
            if self.device == "cuda"
            else "eager",
        )
        self.processor = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    def format_inputs(
        self,
        prompts: List[str],
        images: Optional[List[Image.Image]] = None,
        audios: Optional[List[Audio]] = None,
    ) -> str:
        inputs = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.model_config.enable_thinking,
            )
            for prompt in prompts
        ]
        model_inputs = self.processor(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
        ).to(self.device)
        return model_inputs

    def format_outputs(self, outputs: List[str]) -> List[Tuple[str, str]]:
        if not self.model_config.enable_thinking:
            return [("", output) for output in outputs]
        else:
            return [
                self.parse_thinking_content(output)
                if "</think>" in output
                else ("", output)
                for output in outputs
            ]

    def parse_thinking_content(self, output) -> Tuple[str, str]:
        thinking_content = output.split("</think>")[0]
        final_answer = output.split("</think>")[1]
        return thinking_content.replace("<think>", "").strip(), final_answer.strip()
