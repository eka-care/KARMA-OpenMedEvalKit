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