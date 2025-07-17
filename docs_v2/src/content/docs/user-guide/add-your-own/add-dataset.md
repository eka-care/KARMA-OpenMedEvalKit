---
title: Custom Dataset Integration
---

You can create custom datasets by inheriting from `BaseMultimodalDataset` and implementing the `format_item` method to return a properly formatted `DataLoaderIterable`:

```python
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset
from karma.data_models.dataloader_iterable import DataLoaderIterable

@register_dataset(
    "my_medical_dataset",
    metrics=["exact_match", "accuracy"],
    task_type="mcqa",
    required_args=["split"],
    optional_args=["subset"],
    default_args={"split": "test"}
)
class MyMedicalDataset(BaseMultimodalDataset):
    """Custom medical dataset."""
    
    def __init__(self, split: str = "test", **kwargs):
        self.split = split
        super().__init__(**kwargs)
    
    def load_data(self):
        # Load your dataset
        return your_dataset_loader(split=self.split)
    
    def format_item(self, item):
        """Format each item into DataLoaderIterable format."""
        # Example for text-based dataset
        return DataLoaderIterable(
            input=f"Question: {item['question']}\nChoices: {item['choices']}",
            expected_output=item['answer'],
            other_args={"question_id": item['id']}
        )
```

### Multi-Modal Dataset Example

For datasets that combine multiple modalities:

```python
def format_item(self, item):
    """Format multi-modal item."""
    return DataLoaderIterable(
        input=f"Question: {item['question']}",
        images=[item['image_bytes']],  # List of image bytes
        audio=item.get('audio_bytes'),  # Optional audio
        expected_output=item['answer'],
        other_args={
            "question_type": item['type'],
            "difficulty": item['difficulty']
        }
    )
```

### Conversation Dataset Example

For datasets with multi-turn conversations:

```python
from karma.data_models.dataloader_iterable import Conversation, ConversationTurn

def format_item(self, item):
    """Format conversation item."""
    conversation_turns = []
    for turn in item['conversation']:
        conversation_turns.append(
            ConversationTurn(
                content=turn['content'],
                role=turn['role']  # 'user' or 'assistant'
            )
        )
    
    return DataLoaderIterable(
        conversation=Conversation(conversation_turns=conversation_turns),
        system_prompt=item.get('system_prompt', ''),
        expected_output=item['expected_response']
    )
```

The `DataLoaderIterable` format ensures that all datasets work seamlessly with any model type, whether it's text-only, multi-modal, or conversation-based. Models receive the appropriate data fields and can process them according to their capabilities.

### Local Dataset Example
This example shows how to integrate a local dataset saved, for example, as a Parquet file on disk. It demonstrates loading, formatting, and iterating over the dataset with optional filtering and caching.

```python
import os
import pandas as pd
from typing import Any, Dict, Tuple, Generator, Optional
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset


CONFINEMENT_INSTRUCTIONS = """<Confinement Instructions>"""
SPLIT = "<Split Type>"  # e.g., "test", "train", "validation"
DATASET_NAME = "<Dataset Name>"


@register_dataset(
    dataset_name=DATASET_NAME,
    split=SPLIT,
    metrics=["exact_match"],
    task_type="mcqa",
)
class LocalDataset(BaseMultimodalDataset):
    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        confinement_instructions: str = CONFINEMENT_INSTRUCTIONS,
        **kwargs,
    ):
        # Load local data first
        self.data_path = (
            <paht_to_your_local_dataset>  # e.g., "data/nfi_mcqa.parquet"
        )
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        self.df = pd.read_parquet(self.data_path)

        self.dataset = None
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            confinement_instructions=confinement_instructions,
            **kwargs,
        )

        self.dataset_name = dataset_name
        self.confinement_instructions = confinement_instructions
        self.split = SPLIT
        self.stream = False
        self.processors = kwargs.get("processors", [])
        self.max_samples = kwargs.get("max_samples", None)

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        if self.dataset is None:
            self.dataset = list(self.load_eval_dataset())  # cache once
        for idx, sample in enumerate(self.dataset):
            if self.max_samples is not None and idx >= self.max_samples:
                break
            item = self.format_item(sample)
            yield item

    def __len__(self):
        return len(self.df)

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        input_text = self._format_question(sample["data"])
        correct_answer = sample["data"]["ground_truth"]

        prompt = self.confinement_instructions.replace("<QUESTION>", input_text)

        dataloader_item = DataLoaderIterable(
            input=prompt, expected_output=correct_answer
        )

        dataloader_item.conversation = None

        return dataloader_item

    def extract_prediction(self, response: str) -> Tuple[str, bool]:
        answer, success = "", False
        if "Final Answer:" in response:
            answer = response.split("Final Answer:")[1].strip()
            if answer.startswith("(") and answer.endswith(")"):
                answer = answer[1:-1]
            success = True
        return answer, success

    def _format_question(self, data: Dict[str, Any]) -> str:
        question = data["question"]
        options = data["options"]
        letters = ["A", "B", "C", "D"]
        formatted = [f"{l}. {opt}" for l, opt in zip(letters, options)]
        return f"{question}\n" + "\n".join(formatted)

    def load_eval_dataset(self,
                          dataset_name: str,
                          split: str = "test",
                          config: Optional[str] = None,
                          stream: bool = True,
                          commit_hash: Optional[str] = None,
                          **kwargs):
        for _, row in self.df.iterrows():
            prediction = None
            parsed_output = row.get("model_output_parsed", None)
            if isinstance(parsed_output, dict):
                prediction = parsed_output.get("prediction", None)

            yield {
                "id": row["index"],
                "data": {
                    "question": row["question"],
                    "options": row["options"],
                    "ground_truth": row["ground_truth"],
                },
                "prediction": prediction,
                "metadata": {
                    "generic_name": row.get("generic_name", None),
                    "category": row.get("category", None),
                    "citation": row.get("citation", None),
                },
            }
```
