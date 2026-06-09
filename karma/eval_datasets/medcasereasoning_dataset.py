"""
MedCaseReasoning dataset for clinical diagnostic reasoning evaluation.

Evaluates models on diagnosis prediction and reasoning quality from clinical
case reports using LLM-as-judge metrics.

Paper: arXiv 2505.11733v2 (NeurIPS 2025)
HuggingFace: zou-lab/MedCaseReasoning
"""

import json
import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset

logger = logging.getLogger(__name__)

DATASET_NAME = "zou-lab/MedCaseReasoning"
SPLIT = "test"
COMMIT_HASH = None  # No pinned hash — uses dataset HEAD

# Paper Prompt 6 — inference prompt template (XML-tagged reasoning/answer).
INFERENCE_PROMPT_TEMPLATE = """Read the following case presentation and give the most likely diagnosis.
First, provide your internal reasoning for the diagnosis within the tags <think> ... </think>.
Then, output the final diagnosis (just the name of the disease/entity) within the tags <answer> ... </answer>.

--------------------------------------
CASE PRESENTATION
--------------------------------------
{case_presentation}

--------------------------------------
OUTPUT TEMPLATE
--------------------------------------
<think>
...your internal reasoning for the diagnosis...
</think>
<answer>
...the name of the disease/entity...
</answer>"""

# Template rendered once per few-shot example.
FEW_SHOT_EXAMPLE_TEMPLATE = """--------------------------------------
CASE PRESENTATION
--------------------------------------
{case_presentation}

--------------------------------------
ANSWER
--------------------------------------
<think>
{reasoning}
</think>
<answer>
{diagnosis}
</answer>"""


@register_dataset(
    DATASET_NAME,
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    metrics=[
        "medcasereasoning_diagnostic_accuracy",
        "medcasereasoning_reasoning_recall",
    ],
    task_type="open_ended_qa",
    optional_args=["num_few_shot_examples", "few_shot_seed"],
    default_args={"num_few_shot_examples": 0, "few_shot_seed": 42},
)
class MedCaseReasoningDataset(BaseMultimodalDataset):
    """MedCaseReasoning dataset for clinical diagnostic reasoning evaluation.

    Evaluates models on diagnosis prediction from clinical case presentations,
    with optional few-shot examples sampled from the train split.
    """

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        commit_hash: Optional[str] = COMMIT_HASH,
        num_few_shot_examples: int = 0,
        few_shot_seed: int = 42,
        **kwargs,
    ):
        # CLI args arrive as strings — cast to int.
        if isinstance(num_few_shot_examples, str):
            num_few_shot_examples = int(num_few_shot_examples)
        if isinstance(few_shot_seed, str):
            few_shot_seed = int(few_shot_seed)

        self.num_few_shot_examples = num_few_shot_examples
        self.few_shot_seed = few_shot_seed
        self.few_shot_examples: List[Dict[str, Any]] = []

        # super().__init__ calls load_eval_dataset internally.
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            commit_hash=commit_hash,
            **kwargs,
        )

        # Load few-shot examples from the train split AFTER super().__init__()
        # using a separate load_dataset call (not load_eval_dataset, which would
        # overwrite self.dataset).
        if self.num_few_shot_examples > 0:
            self._load_few_shot_examples()

    def _load_few_shot_examples(self) -> None:
        """Load few-shot examples from the train split."""
        try:
            train_ds = load_dataset(
                DATASET_NAME, split="train", streaming=False
            )
            rng = random.Random(self.few_shot_seed)
            n = min(self.num_few_shot_examples, len(train_ds))
            indices = rng.sample(range(len(train_ds)), n)
            self.few_shot_examples = [train_ds[i] for i in indices]
            logger.info(f"Loaded {len(self.few_shot_examples)} few-shot examples")
        except Exception as e:
            logger.warning(
                f"Failed to load few-shot examples: {e}. Falling back to 0-shot."
            )
            self.few_shot_examples = []

    def _build_few_shot_prefix(self) -> str:
        """Build few-shot prefix string from loaded examples."""
        if not self.few_shot_examples:
            return ""

        parts = ["Here are some example case presentations with their diagnoses:\n"]
        for i, ex in enumerate(self.few_shot_examples, 1):
            parts.append(
                f"Example {i}:\n"
                + FEW_SHOT_EXAMPLE_TEMPLATE.format(
                    case_presentation=ex["case_prompt"],
                    reasoning=ex.get("diagnostic_reasoning") or "",
                    diagnosis=ex["final_diagnosis"],
                )
            )
        parts.append("\nNow answer the following case:\n")
        return "\n".join(parts)

    def _parse_reasoning_steps(self, reasoning_text: str) -> List[str]:
        """Parse numbered reasoning steps from the ``diagnostic_reasoning`` field.

        The field is formatted as ``"1. [text] 2. [text] ..."`` (sometimes
        newline-separated, sometimes inline). Returns a list of step strings;
        empty list for empty/None input.
        """
        if not reasoning_text:
            return []

        # Split on newline OR a space preceding "N." so we handle both inline
        # and newline-separated variants.
        parts = re.split(r"(?:\n|(?<=\S) )(?=\d+\.)", reasoning_text.strip())
        steps = []
        for part in parts:
            cleaned = re.sub(r"^\d+\.\s*", "", part.strip())
            if cleaned:
                steps.append(cleaned)
        return steps

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        """Format one HF row into a DataLoaderIterable.

        The per-sample reasoning steps are JSON-serialised into
        ``other_args["reasoning_steps"]`` so the ``reasoning_recall`` metric
        can reconstruct them downstream without re-parsing.
        """
        case_prompt = sample["case_prompt"]
        final_diagnosis = sample["final_diagnosis"]
        diagnostic_reasoning = sample.get("diagnostic_reasoning")

        reasoning_steps_json = json.dumps(
            self._parse_reasoning_steps(diagnostic_reasoning)
        )

        prompt = self._build_few_shot_prefix() + INFERENCE_PROMPT_TEMPLATE.format(
            case_presentation=case_prompt
        )

        return DataLoaderIterable(
            input=prompt,
            expected_output=final_diagnosis,
            other_args={"reasoning_steps": reasoning_steps_json},
        )

    def extract_prediction(self, response: str) -> Tuple[str, bool]:
        """Extract the prediction from a model response.

        CRITICAL: returns the FULL response rather than just the ``<answer>``
        contents because the ``reasoning_recall`` metric needs the ``<think>``
        block as well. The ``success`` bool indicates whether the expected
        ``<answer>`` tag was present.
        """
        match = re.search(
            r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE
        )
        if match:
            return (response.strip(), True)
        return (response.strip(), False)
