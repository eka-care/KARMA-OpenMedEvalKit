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

### Using Local Datasets with KARMA
This guide will walk you through how to plug that dataset into KARMA’s evaluation pipeline.

#### Step 1: Organize Your Dataset
Ensure your dataset is structured correctly. 

Each row should ideally include:

- A question

- A list of options (for multiple-choice) (Optional)

- The correct answer

- Optionally: metadata like category, generic name, or citation

#### Step 2: Set Up a Custom Dataset Class
KARMA supports registering your own datasets using a decorator.

```python
@register_dataset(
    dataset_name="nfi-mcqa-local",
    split="test",
    metrics=["exact_match"],
    task_type="mcqa",  # Multiple Choice Q&A
)
class LocalDataset(BaseMultimodalDataset):
    ...
```

This decorator registers your dataset with KARMA, allowing it to be used in evaluations. 

The `task_type` should match the type of task your dataset is designed for (e.g., `mcqa` for multiple-choice question answering).

#### Step 3: Load your Dataset
In your Dataset class, load your dataset file. You can use any format supported by pandas, such as CSV or Parquet.
```python
def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        confinement_instructions: str = CONFINEMENT_INSTRUCTIONS,
        **kwargs,
    ):
        # Load local data first
        self.data_path = (
            <path_to_your_dataset>  # e.g., "data/nfi_mcqa.parquet"
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

```

#### Step 4: Implement the `format_item` Method
Each row in your dataset will eventually be converted into an input-output pair for the model. 

```python
def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
    input_text = self._format_question(sample["data"])
    correct_answer = sample["data"]["ground_truth"]

    prompt = self.confinement_instructions.replace("<QUESTION>", input_text)

    dataloader_item = DataLoaderIterable(
        input=prompt, expected_output=correct_answer
    )

    dataloader_item.conversation = None

    return dataloader_item
```
This method should return a `DataLoaderIterable` object containing the formatted input and expected output for each sample.

#### 5: Iterate Over the Dataset
KARMA will use __iter__() to go through your dataset, so we implement it to yield formatted examples:
```python
def __iter__(self) -> Generator[Dict[str, Any], None, None]:
    if self.dataset is None:
        self.dataset = list(self.load_eval_dataset())  # cache once
    for idx, sample in enumerate(self.dataset):
        if self.max_samples is not None and idx >= self.max_samples:
            break
        item = self.format_item(sample)
        yield item
```

#### Step 6: Handle Model Output
To extract the model's predictions, implement the `extract_prediction` method. 
```python
def extract_prediction(self, response: str) -> Tuple[str, bool]:
    answer, success = "", False
    if "Final Answer:" in response:
        answer = response.split("Final Answer:")[1].strip()
        if answer.startswith("(") and answer.endswith(")"):
            answer = answer[1:-1]
        success = True
    return answer, success
```

#### Step 7: Yield Examples for Evaluation
Finally, load_eval_dataset reads from your DataFrame and returns structured examples.
```python
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
If your dataset already includes previous model outputs, this is where you can plug them in.
