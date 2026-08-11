
import logging
from typing import Any, Dict, Tuple

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.eval_datasets.extraction_utils import extract_wrapped_mcq_answer
from karma.registries.dataset_registry import register_dataset

logger = logging.getLogger(__name__)

DATASET_NAME = "ekacare/mmlu-medical-mcqs-evaluation-dataset"
SPLIT = "test"

# Available subsets (12 total)
VALID_SUBSETS = [
    "anatomy",
    "clinical-knowledge",
    "college-biology",
    "college-chemistry",
    "college-medicine",
    "high-school-biology",
    "human-aging",
    "medical-genetics",
    "nutrition",
    "professional-medicine",
    "professional-psychology",
    "virology",
]

@register_dataset(
    DATASET_NAME,
    split=SPLIT,
    required_args=["subset"],
    metrics=["exact_match"],
    task_type="mcqa",
    default_args={"user_prompt": "eka_mmlu_medical_cot"},
)
class EkaMMLUMedicalDataset(BaseMultimodalDataset):
    """
    Eka MMLU Medical MCQs Evaluation Dataset.
    
    A unified dataset for MMLU medical subsets supporting subset-level accuracy.
    
    Available subsets:
        - anatomy
        - clinical-knowledge
        - college-biology
        - college-chemistry
        - college-medicine
        - high-school-biology
        - human-aging
        - medical-genetics
        - nutrition
        - professional-medicine
        - professional-psychology
        - virology
    """

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        subset: str = "anatomy",
        **kwargs,
    ):
        """
        Initialize Eka MMLU Medical dataset.

        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split (default: test)
            subset: One of the 12 medical subsets
            **kwargs: Additional arguments passed to base class
        """
        if subset not in VALID_SUBSETS:
            raise ValueError(
                f"Invalid subset '{subset}'. Must be one of: {VALID_SUBSETS}"
            )

        self.subset = subset
        # Update dataset_name to include subset for unique identification
        self.dataset_name = f"{DATASET_NAME}-{self.subset}"

        super().__init__(
            dataset_name=dataset_name,
            split=split,
            config=self.subset,
            **kwargs,
        )
        
        logger.info(f"Loaded EkaMMLUMedicalDataset (subset: {subset})")

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        """
        Format a sample into a text prompt.

        Args:
            sample: A single sample from the dataset with fields:
                - centerpiece: The question text
                - options: List of 4 answer options
                - correct_options: List with correct letter (e.g., ['B'])

        Returns:
            DataLoaderIterable with formatted prompt and expected output
        """
        question = sample["centerpiece"]
        options = sample["options"]
        
        formatted_choices = []
        choice_labels = ["A", "B", "C", "D"]
        for i, option in enumerate(options):
            if i < len(choice_labels):
                formatted_choices.append(f"{choice_labels[i]}. {option}")
        
        formatted_question = f"{question}\n" + "\n".join(formatted_choices)
        prompt = self.confinement_instructions.replace("<QUESTION>", formatted_question)
        
        correct_answer = sample["correct_options"][0] if sample["correct_options"] else ""
        
        return DataLoaderIterable(
            input=prompt,
            expected_output=correct_answer,
        )

    def extract_prediction(self, response: str, **kwargs) -> Tuple[str, bool]:
        """
        Extract answer from model response.

        Args:
            response: Model's response text

        Returns:
            Tuple of (extracted_answer, success_flag)
        """
        answer, success = extract_wrapped_mcq_answer(response, valid_letters="ABCD")
        if not answer:
            logger.warning(f"No answer found in response: {response[:100]}...")
        return answer, success

