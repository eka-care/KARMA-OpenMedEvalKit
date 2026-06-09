"""
AfriMed-QA dataset implementation for KARMA.

This module provides dataset classes for the AfriMed-QA v2 benchmark:
- AfriMedQAConsumerQueriesDataset: Open-ended consumer health questions (no reference answers)
- AfriMedQAMCQDataset: Multiple choice medical questions with answers

Dataset: https://huggingface.co/datasets/intronhealth/afrimedqa_v2
Paper: AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset

The dataset contains 15,275 questions from 621 contributors across 16 African countries,
covering 32+ medical specialties.
"""

import json
import logging
from typing import Dict, Any, Tuple, Optional, List

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset

logger = logging.getLogger(__name__)

# Dataset Configuration
DATASET_NAME = "intronhealth/afrimedqa_v2"
SPLIT = "train"  # The HuggingFace split (dataset uses internal 'split' field for train/test)
COMMIT_HASH = None  # Use latest version

# Confinement instructions for Consumer Queries (open-ended)
CONSUMER_QUERIES_INSTRUCTIONS = """You are a medical assistant helping answer health questions from African healthcare consumers.

Clinical Context: <CONTEXT>

Question: <QUESTION>

Provide a helpful, accurate, and medically sound response. Your answer should be informative yet accessible to a general audience."""

# Confinement instructions for MCQ
MCQ_CONFINEMENT_INSTRUCTIONS = """
Read the question carefully and select the most appropriate answer.

Question: <QUESTION>

Options:
<OPTIONS>

Response in the following format - "A", no additional reasoning or justification needed."""


@register_dataset(
    DATASET_NAME,
    split=SPLIT,
    commit_hash=COMMIT_HASH,
    metrics=["tokenised_f1", "bleu", "rouge"],
    task_type="open_ended_qa",
    optional_args=["confinement_instructions", "internal_split"],
    default_args={"internal_split": "test"},
)
class AfriMedQAConsumerQueriesDataset(BaseMultimodalDataset):
    """
    AfriMed-QA Consumer Queries Dataset for open-ended medical QA.

    This dataset contains 10,000 open-ended medical questions from African healthcare
    consumers. Note: This subset does NOT have reference answers, making it suitable
    for:
    - Qualitative evaluation
    - Rubric-based evaluation
    - Human evaluation
    - LLM-as-judge evaluation

    The metrics (tokenised_f1, bleu, rouge) can be used if you provide your own
    reference answers or use the 'prompt' field (clinical scenario) as context.

    Attributes:
        question_type: Always 'consumer_queries'
        internal_split: 'train' (8,366 samples) or 'test' (1,634 samples)
    """

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        commit_hash: Optional[str] = COMMIT_HASH,
        confinement_instructions: str = CONSUMER_QUERIES_INSTRUCTIONS,
        internal_split: str = "test",
        **kwargs,
    ):
        """
        Initialize AfriMed-QA Consumer Queries dataset.

        Args:
            dataset_name: HuggingFace dataset identifier
            split: HuggingFace dataset split (always 'train' for this dataset)
            commit_hash: Specific commit hash for reproducibility
            confinement_instructions: Instructions template for the model
            internal_split: Internal data split - 'train' or 'test'
            **kwargs: Additional arguments passed to base class
        """
        self.confinement_instructions = confinement_instructions
        self.internal_split = internal_split

        super().__init__(
            dataset_name=dataset_name,
            split=split,
            commit_hash=commit_hash,
            **kwargs,
        )

        # Filter for Consumer Queries only and internal split
        self.dataset = self.dataset.filter(
            lambda x: (
                x.get("question_type", "").lower().replace(" ", "_")
                == "consumer_queries"
                and x.get("split", "") == self.internal_split
            )
        )

        logger.info(
            f"AfriMed-QA Consumer Queries dataset loaded "
            f"(internal_split={self.internal_split})"
        )

    def format_item(self, sample: Dict[str, Any], **kwargs) -> DataLoaderIterable:
        """
        Format a sample into a prompt for open-ended medical QA.

        Args:
            sample: A single sample from the dataset

        Returns:
            DataLoaderIterable with formatted prompt
        """
        # Use cleaned question if available
        question = sample.get("question_clean") or sample.get("question", "")

        # Clinical context/scenario
        context = sample.get("prompt", "")

        # Build the prompt
        prompt = self.confinement_instructions
        prompt = prompt.replace(
            "<CONTEXT>", context if context else "No additional context provided."
        )
        prompt = prompt.replace("<QUESTION>", question)

        # Store metadata for analysis
        other_args = {
            "sample_id": sample.get("sample_id", ""),
            "specialty": sample.get("specialty", ""),
            "country": sample.get("country", ""),
            "tier": sample.get("tier", ""),
            "region_specific": sample.get("region_specific", False),
            "clinical_scenario": context,
        }

        # Note: Consumer queries don't have reference answers
        # Using the question as expected_output for basic tracking
        # For proper evaluation, use rubric-based or LLM-as-judge approaches
        return DataLoaderIterable(
            input=prompt,
            expected_output=question,  # No reference answer available
            other_args=other_args,
            **kwargs,
        )

    def extract_prediction(self, response: str, **kwargs) -> Tuple[str, bool]:
        """Extract prediction from model response."""
        prediction = response.strip()
        return prediction, bool(prediction)


# Create a unique identifier for MCQ dataset
MCQ_DATASET_IDENTIFIER = f"{DATASET_NAME}/mcq"


@register_dataset(
    MCQ_DATASET_IDENTIFIER,
    split=SPLIT,
    commit_hash=COMMIT_HASH,
    metrics=["exact_match", "geographic_accuracy", "demographic_accuracy"],
    task_type="mcqa",
    optional_args=["confinement_instructions", "internal_split", "specialty"],
    default_args={"internal_split": "test", "specialty": None},
)
class AfriMedQAMCQDataset(BaseMultimodalDataset):
    """
    AfriMed-QA Multiple Choice Questions Dataset.

    This dataset contains expert-generated multiple choice medical questions
    with verified correct answers. Total 3,766 MCQ questions:
    - Train: 129 samples
    - Test: 3,637 samples

    All questions have:
    - correct_answer: The correct option (e.g., "option1", "option2", etc.)
    - answer_options: List of answer choices
    - specialty: Medical specialty
    - answer_rationale: Explanation for the correct answer (some samples)
    """

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        commit_hash: Optional[str] = COMMIT_HASH,
        confinement_instructions: str = MCQ_CONFINEMENT_INSTRUCTIONS,
        internal_split: str = "test",
        specialty: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize AfriMed-QA MCQ dataset.

        Args:
            dataset_name: HuggingFace dataset identifier
            split: HuggingFace dataset split
            commit_hash: Specific commit hash for reproducibility
            confinement_instructions: Instructions template for the model
            internal_split: Internal data split - 'train' or 'test'
            specialty: Optional filter by medical specialty
            **kwargs: Additional arguments passed to base class
        """
        self.internal_split = internal_split
        self.specialty_filter = specialty

        super().__init__(
            dataset_name=dataset_name,
            split=split,
            commit_hash=commit_hash,
            confinement_instructions=confinement_instructions,
            **kwargs,
        )

        # Filter for MCQ questions and internal split
        self.dataset = self.dataset.filter(
            lambda x: (
                x.get("question_type", "").lower() == "mcq"
                and x.get("split", "") == self.internal_split
                and x.get("tier", "") == "expert"
            )
        )

        # Optional specialty filter
        if self.specialty_filter:
            self.dataset = self.dataset.filter(
                lambda x: x.get("specialty") == self.specialty_filter
            )
            logger.info(f"Filtered to specialty: {self.specialty_filter}")

        logger.info(
            f"AfriMed-QA MCQ dataset loaded (internal_split={self.internal_split})"
        )

    def format_item(self, sample: Dict[str, Any], **kwargs) -> DataLoaderIterable:
        """
        Format a sample into a MCQ prompt.

        Args:
            sample: A single sample from the dataset

        Returns:
            DataLoaderIterable with formatted prompt and expected answer
        """
        # Get question text
        question = sample.get("question_clean")
        if question is None or question == "":
            question = sample.get("question")

        # Get answer options and format them
        answer_options_raw = sample.get("answer_options", [])
        if answer_options_raw is None:
            answer_options_raw = []

        # Parse JSON string if needed (HuggingFace returns it as JSON string)
        if isinstance(answer_options_raw, str):
            try:
                answer_options_raw = json.loads(answer_options_raw)
            except json.JSONDecodeError:
                answer_options_raw = []

        # Convert dict format {"option1": "...", "option2": "..."} to list
        if isinstance(answer_options_raw, dict):
            # Sort by option key to maintain order (option1, option2, etc.)
            sorted_keys = sorted(answer_options_raw.keys())
            answer_options = [
                answer_options_raw[k]
                for k in sorted_keys
                if answer_options_raw[k] != "n/a"
            ]
        else:
            answer_options = answer_options_raw

        # Format options with letters
        choice_labels = ["A", "B", "C", "D", "E"]
        formatted_options = []
        for i, option in enumerate(answer_options):
            if i < len(choice_labels):
                formatted_options.append(f"{choice_labels[i]}. {option}")

        options_text = "\n".join(formatted_options)

        # Get correct answer and convert to letter
        correct_answer_raw = sample.get("correct_answer", "")
        correct_answer = self._convert_answer_to_letter(
            correct_answer_raw, answer_options
        )

        # Build the prompt
        prompt = self.confinement_instructions
        prompt = prompt.replace("<QUESTION>", question)
        prompt = prompt.replace("<OPTIONS>", options_text)

        # Store metadata for analysis and stratified metrics
        other_args = {
            "sample_id": sample.get("sample_id", ""),
            "specialty": sample.get("specialty", ""),
            "country": sample.get("country", ""),
            "answer_rationale": sample.get("answer_rationale", ""),
            "tier": sample.get("tier", ""),
            "raw_answer": correct_answer_raw,
            "answer_options": answer_options,
            # Demographic fields for stratified accuracy metrics
            "mentions_gender": sample.get("mentions_gender"),
            "mentions_age": sample.get("mentions_age"),
        }

        return DataLoaderIterable(
            input=prompt,
            expected_output=correct_answer,
            other_args=other_args,
            **kwargs,
        )

    def _convert_answer_to_letter(
        self, answer_raw: str, answer_options: List[str]
    ) -> str:
        """
        Convert raw answer format to letter (A, B, C, D, E).

        Args:
            answer_raw: Raw answer like "option1", "option2", or the actual text
            answer_options: List of answer options

        Returns:
            Letter corresponding to the correct answer
        """
        choice_labels = ["A", "B", "C", "D", "E"]

        if not answer_raw:
            return ""

        # Handle "option1", "option2", etc. format
        if answer_raw.lower().startswith("option"):
            try:
                option_num = int(answer_raw.lower().replace("option", "")) - 1
                if 0 <= option_num < len(choice_labels):
                    return choice_labels[option_num]
            except ValueError:
                pass

        # Handle direct text match
        answer_lower = answer_raw.lower().strip()
        for i, option in enumerate(answer_options):
            if option and option.lower().strip() == answer_lower:
                if i < len(choice_labels):
                    return choice_labels[i]

        # Default: return first letter if nothing matches
        logger.warning(f"Could not convert answer '{answer_raw}' to letter")
        return answer_raw

    def extract_prediction(self, response: str, **kwargs) -> Tuple[str, bool]:
        """
        Extract MCQ prediction from model response.

        Args:
            response: Raw model output

        Returns:
            Tuple of (extracted_letter, success_flag)
        """
        import re

        answer = ""
        success = False
        valid_letters = ["A", "B", "C", "D", "E"]

        # Helper to clean and extract letter from short text (1-5 chars)
        def extract_letter_short(text: str) -> Optional[str]:
            # For short strings like "D?", "B)", "(C)" - remove non-letters
            cleaned = re.sub(r"[^a-zA-Z]", "", text).upper()
            if len(cleaned) == 1 and cleaned in valid_letters:
                return cleaned
            return None

        # Try to extract from "Final Answer:" format (case-insensitive)
        # Handles: "Final Answer: (B)", "Final Answer: B", "Final Answer (B)", etc.
        final_answer_match = re.search(
            r"final\s*answer\s*[:\-]?\s*[([]?\s*([a-eA-E])\s*[)\]]?",
            response,
            re.IGNORECASE,
        )
        if final_answer_match:
            answer = final_answer_match.group(1).upper()
            success = True

        # Fallback: look for "answer is X" or "answer: X" patterns
        if not success:
            answer_pattern = re.search(
                r"answer\s*(?:is|:)\s*\(?([a-eA-E])\)?", response, re.IGNORECASE
            )
            if answer_pattern:
                answer = answer_pattern.group(1).upper()
                success = True

        # Fallback: look for patterns like "(D)", "D)", "D.", "d?" at the end of last line
        if not success:
            lines = response.strip().split("\n")
            last_line = lines[-1].strip()

            # Try to extract single letter from short last line
            if len(last_line) <= 5:
                letter = extract_letter_short(last_line)
                if letter:
                    answer = letter
                    success = True

        # Fallback: find letter followed by punctuation at end of response
        if not success:
            end_pattern = re.search(r"([a-eA-E])\s*[).?!]?\s*$", response.strip())
            if end_pattern:
                answer = end_pattern.group(1).upper()
                success = True

        # Last resort: find any isolated letter (with word boundaries) near the end
        if not success:
            # Look for standalone letter in last 50 chars
            last_chunk = response.strip()[-50:]
            isolated_letter = re.search(r"\b([a-eA-E])\b", last_chunk)
            if isolated_letter:
                answer = isolated_letter.group(1).upper()
                success = True

        if not success:
            logger.warning(f"Could not extract MCQ answer from: {response[:100]}...")
            # Return raw response when extraction fails so it's visible in cache
            return response.strip(), False

        return answer, success
