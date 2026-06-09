"""
MedMCQA dataset implementation with multimodal support.

This module provides the MedMCQADataset class that implements the new
multimodal dataset interface for use with the refactored benchmark system.
"""

import logging
from typing import Dict, Any, Tuple

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.eval_datasets.extraction_utils import extract_wrapped_mcq_answer
from karma.registries.dataset_registry import register_dataset

logger = logging.getLogger(__name__)

# Hardcoded confinement instructions
CONFINEMENT_INSTRUCTIONS = """Instructions: The following are multiple choice questions about medical knowledge. Solve them in a
step-by-step fashion, starting by summarizing the available information. Output a single option from the
four options as the final answer. Question: <QUESTION> Response (think step by step and then
end with "Final Answer:" followed by *only* the letter corresponding to the correct answer enclosed in
parentheses)"""
DATASET_NAME = "openlifescienceai/medmcqa"
SPLIT = "validation"
COMMIT_HASH = "91c6572c454088bf71b679ad90aa8dffcd0d5868"


@register_dataset(
    DATASET_NAME,
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    metrics=["exact_match"],
    task_type="mcqa",
)
class MedMCQADataset(BaseMultimodalDataset):
    """
    MedMCQA PyTorch Dataset implementing the new multimodal interface.
    """

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        confinement_instructions: str = CONFINEMENT_INSTRUCTIONS,
        commit_hash: str = COMMIT_HASH,
        **kwargs,
    ):
        """
        Initialize MedMCQA dataset.

        Args:
            dataset_name: HuggingFace dataset identifier (defaults to
                ``openlifescienceai/medmcqa``).
            split: dataset split name (defaults to ``validation``).
            confinement_instructions: prompt template wrapping the question.
                Exposed as a keyword argument (rather than assigned directly
                on ``self`` before ``super().__init__``) so that the value is
                correctly forwarded into ``BaseMultimodalDataset.__init__``.
            commit_hash: HuggingFace dataset revision pin.
            **kwargs: Additional arguments passed to base class.

        Notes:
            -----------------------------------------------------------------
            Confinement-instructions overwrite bug (fixed 2026-04-14)
            -----------------------------------------------------------------
            BUG (pre-fix): the previous implementation set
            ``self.confinement_instructions = CONFINEMENT_INSTRUCTIONS`` on
            the child instance BEFORE calling ``super().__init__(...)`` and
            did NOT forward the value to the super call. That looked correct
            but silently clobbered the child's assignment because
            ``BaseMultimodalDataset.__init__`` (see
            ``karma/eval_datasets/base_dataset.py:59``) unconditionally runs
            ``self.confinement_instructions = confinement_instructions``,
            where the ``confinement_instructions`` parameter defaults to the
            empty string ``""``. Net effect: the child's pre-assignment was
            overwritten to ``""`` the moment control passed into the parent
            constructor, and every downstream call to
            ``self.confinement_instructions.replace("<QUESTION>", input_text)``
            produced an empty prompt. The empty prompt propagated through the
            benchmark loop and crashed in ``karma/benchmark.py:581`` with an
            ``AttributeError: 'NoneType' object has no attribute 'get'``
            because the empty prompt caused a ``None`` to appear in the
            ``model_results`` list returned by ``batch_predict``.

            WHY THIS FIX WORKS: by accepting ``confinement_instructions`` as
            a keyword argument in this ``__init__`` signature and forwarding
            it into ``super().__init__(confinement_instructions=...)``, the
            parent constructor receives the correct template string and its
            unconditional ``self.confinement_instructions = ...`` assignment
            now lands on the intended value rather than the empty-string
            default. This mirrors the clean pattern already used by
            ``karma/eval_datasets/medqa_dataset.py`` (the authoritative
            reference pattern confirmed working end-to-end by sub-agent A).

            Reference: the bug was identified and the fix recommended by
            Phase 2.5 sub-agent A in
            ``.claude/reports/clinical_ai_implementation/phase3_q3q5_investigation_handoff.md``
            (Q3 section). Sub-agent A additionally verified that the three
            sibling MCQA datasets (``medqa``, ``medqa_usmle``,
            ``eka_mmlu_medical``) had already been remediated upstream via
            either this clean pattern or the belt-and-suspenders pattern
            (pre-assign AND forward to super), while ``medmcqa`` was the
            last remaining instance of the buggy pattern.
        """
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            confinement_instructions=confinement_instructions,
            commit_hash=commit_hash,
            **kwargs,
        )

    def format_item(self, sample: Dict[str, Any], **kwargs):
        """
        Format a sample into a text prompt for MedMCQA.

        Args:
            sample: A single sample from the dataset

        Returns:
            Formatted text prompt
        """
        input_text = self._format_question(sample)

        # Parse correct answer from cop field
        cop = sample["cop"]
        choice_labels = ["A", "B", "C", "D"]
        correct_answer = choice_labels[cop]
        prompt = self.confinement_instructions.replace("<QUESTION>", input_text)
        processed_sample = DataLoaderIterable(
            input=prompt, expected_output=correct_answer, **kwargs
        )

        # Add confinement instructions to the question and options

        return processed_sample

    def extract_prediction(self, response: str, **kwargs) -> Tuple[str, bool]:
        answer, success = extract_wrapped_mcq_answer(response, valid_letters="ABCD")
        if not answer:
            logger.warning(f"No answer found in response: {response}")
        return answer, success

    def _format_question(self, data: Dict[str, Any]) -> str:
        """
        Format a single MedMCQA question.

        Args:
            data: Dictionary containing question data with keys:
                - question: The question text
                - opa, opb, opc, opd: The four answer choices
                - cop: The correct answer (if include_answer is True)
            include_answer: Whether to include the answer in the formatted question

        Returns:
            Formatted question string
        """
        question = data["question"]
        choices = [data["opa"], data["opb"], data["opc"], data["opd"]]

        # Format choices as A, B, C, D
        formatted_choices = []
        choice_labels = ["A", "B", "C", "D"]
        for i, choice in enumerate(choices):
            formatted_choices.append(f"{choice_labels[i]}. {choice}")

        formatted_question = f"{question}\n" + "\n".join(formatted_choices)

        return formatted_question
