---
title: Adding a New Model
---

This guide provides a comprehensive walkthrough for adding new models to the KARMA evaluation framework. KARMA supports diverse model types including local HuggingFace models, API-based services, and multi-modal models across text, audio, image, and video domains.

## Architecture Overview

### Base Model System

All models in KARMA inherit from the `BaseModel` abstract class, which provides a unified interface for model loading, inference, and data processing.

```python
from karma.models.base_model_abs import BaseModel
from karma.data_models.dataloader_iterable import DataLoaderIterable

class MyModel(BaseModel):
    def load_model(self):
        """Initialize model and tokenizer/processor"""
        pass
    
    def run(self, inputs: List[DataLoaderIterable]) -> List[str]:
        """Main inference method"""
        pass
    
    def preprocess(self, inputs: List[DataLoaderIterable]) -> Any:
        """Convert raw inputs to model-ready format"""
        pass
    
    def postprocess(self, outputs: Any) -> List[str]:
        """Process model outputs to final format"""
        pass
```

### ModelMeta System

The `ModelMeta` class provides comprehensive metadata management for model registration:

```python
from karma.data_models.model_meta import ModelMeta, ModelType, ModalityType

model_meta = ModelMeta(
    name="my-model/my-model-name",
    description="Description of my model",
    loader_class="karma.models.my_model.MyModel",
    loader_kwargs={
        "temperature": 0.7,
        "max_tokens": 2048,
    },
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    framework=["PyTorch", "Transformers"],
)
```

### Data Flow

Models process data through the `DataLoaderIterable` structure.
This object is passed from the benchmark class to the model through the huggingface `load_dataset` method.

```python
from karma.data_models.dataloader_iterable import DataLoaderIterable

# Input data structure
data = DataLoaderIterable(
    input="Your text input here",
    images=None,  # PIL Images or bytes
    audio=None,   # Audio data
    conversation=None,  # Multi-turn conversations
    system_prompt="System instructions",
    expected_output="Ground truth for evaluation",
    other_args={"custom_key": "custom_value"}
)
```

## Model Implementation Steps

### Step 1: Create Model Class

Create a new Python file in the `karma/models/` directory:

```python
# karma/models/my_model.py
import torch
from typing import List, Dict, Any
from karma.models.base_model_abs import BaseModel
from karma.data_models.dataloader_iterable import DataLoaderIterable

class MyModel(BaseModel):
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 2048)
        
    def load_model(self):
        """Load the model and tokenizer"""
        # Example for HuggingFace model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True
        )
        
        self.is_loaded = True
        
    def preprocess(self, inputs: List[DataLoaderIterable]) -> Dict[str, torch.Tensor]:
        """Convert inputs to model format"""
        batch_inputs = []
        
        for item in inputs:
            # Handle different input types
            if item.conversation:
                # Multi-turn conversation
                messages = item.conversation.messages
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                # Single input
                text = item.input
                
            batch_inputs.append(text)
        
        # Tokenize batch
        encoding = self.tokenizer(
            batch_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_tokens
        )
        
        return encoding.to(self.device)
    
    def run(self, inputs: List[DataLoaderIterable]) -> List[str]:
        """Generate model outputs"""
        if not self.is_loaded:
            self.load_model()
            
        # Preprocess inputs
        model_inputs = self.preprocess(inputs)
        
        # Generate outputs
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode outputs
        generated_texts = []
        for i, output in enumerate(outputs):
            # Remove input tokens from output
            input_length = model_inputs["input_ids"][i].shape[0]
            generated_tokens = output[input_length:]
            
            text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            generated_texts.append(text)
        
        return self.postprocess(generated_texts)
    
    def postprocess(self, outputs: List[str]) -> List[str]:
        """Clean up generated outputs"""
        cleaned_outputs = []
        for output in outputs:
            # Remove any unwanted tokens or formatting
            cleaned = output.strip()
            cleaned_outputs.append(cleaned)
        
        return cleaned_outputs
```

### Step 2: Create ModelMeta Configuration

Add ModelMeta definitions at the end of your model file:

```python
# karma/models/my_model.py (continued)
from karma.registries.model_registry import register_model_meta
from karma.data_models.model_meta import ModelMeta, ModelType, ModalityType

# Define model variants
MyModelSmall = ModelMeta(
    name="my-org/my-model-small",
    description="Small version of my model",
    loader_class="karma.models.my_model.MyModel",
    loader_kwargs={
        "temperature": 0.7,
        "max_tokens": 2048,
    },
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    framework=["PyTorch", "Transformers"],
    n_parameters=7_000_000_000,
    memory_usage_mb=14_000,
)

MyModelLarge = ModelMeta(
    name="my-org/my-model-large",
    description="Large version of my model",
    loader_class="karma.models.my_model.MyModel",
    loader_kwargs={
        "temperature": 0.7,
        "max_tokens": 4096,
    },
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    framework=["PyTorch", "Transformers"],
    n_parameters=70_000_000_000,
    memory_usage_mb=140_000,
)

# Register models
register_model_meta(MyModelSmall)
register_model_meta(MyModelLarge)
```

### Step 3: Verify Registration

Test that your model is properly registered:

```bash
# List all models to verify registration
karma list models

# Check specific model details
karma list models --name "my-org/my-model-small"
```

## Model Types and Examples

### Text Generation Models

**HuggingFace Transformers Model:**
```python
class HuggingFaceTextModel(BaseModel):
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.is_loaded = True
    
    def run(self, inputs: List[DataLoaderIterable]) -> List[str]:
        # Implementation similar to Step 1 example
        pass
```

**API-Based Model:**
```python
class APITextModel(BaseModel):
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        self.api_key = kwargs.get("api_key")
        self.base_url = kwargs.get("base_url")
        
    def load_model(self):
        import openai
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.is_loaded = True
    
    def run(self, inputs: List[DataLoaderIterable]) -> List[str]:
        if not self.is_loaded:
            self.load_model()
            
        responses = []
        for item in inputs:
            response = self.client.chat.completions.create(
                model=self.model_name_or_path,
                messages=[{"role": "user", "content": item.input}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            responses.append(response.choices[0].message.content)
        
        return responses
```

### Audio Recognition Models

```python
class AudioRecognitionModel(BaseModel):
    def load_model(self):
        import whisper
        self.model = whisper.load_model(self.model_name_or_path)
        self.is_loaded = True
    
    def preprocess(self, inputs: List[DataLoaderIterable]) -> List[Any]:
        audio_data = []
        for item in inputs:
            if item.audio:
                audio_data.append(item.audio)
            else:
                raise ValueError("Audio data is required for audio recognition")
        return audio_data
    
    def run(self, inputs: List[DataLoaderIterable]) -> List[str]:
        if not self.is_loaded:
            self.load_model()
            
        audio_data = self.preprocess(inputs)
        transcriptions = []
        
        for audio in audio_data:
            result = self.model.transcribe(audio)
            transcriptions.append(result["text"])
        
        return transcriptions
```

### Multi-Modal Models

```python
class MultiModalModel(BaseModel):
    def load_model(self):
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name_or_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16
        )
        self.is_loaded = True
    
    def preprocess(self, inputs: List[DataLoaderIterable]) -> Dict[str, torch.Tensor]:
        batch_inputs = []
        
        for item in inputs:
            # Handle text + image inputs
            if item.images and item.input:
                batch_inputs.append({
                    "text": item.input,
                    "images": item.images
                })
            else:
                raise ValueError("Both text and images are required")
        
        # Process with multi-modal processor
        processed = self.processor(
            text=[item["text"] for item in batch_inputs],
            images=[item["images"] for item in batch_inputs],
            return_tensors="pt",
            padding=True
        )
        
        return processed.to(self.device)
    
    def run(self, inputs: List[DataLoaderIterable]) -> List[str]:
        if not self.is_loaded:
            self.load_model()
            
        model_inputs = self.preprocess(inputs)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature
            )
        
        # Decode outputs
        generated_texts = self.processor.batch_decode(
            outputs, skip_special_tokens=True
        )
        
        return generated_texts
```

### ModelMeta Examples for Different Types

```python
# Text generation model
TextModelMeta = ModelMeta(
    name="my-org/text-model",
    loader_class="karma.models.my_model.HuggingFaceTextModel",
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    framework=["PyTorch", "Transformers"],
)

# Audio recognition model
AudioModelMeta = ModelMeta(
    name="my-org/audio-model",
    loader_class="karma.models.my_model.AudioRecognitionModel",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    framework=["PyTorch", "Whisper"],
    audio_sample_rate=16000,
    supported_audio_formats=["wav", "mp3", "flac"],
)

# Multi-modal model
MultiModalMeta = ModelMeta(
    name="my-org/multimodal-model",
    loader_class="karma.models.my_model.MultiModalModel",
    model_type=ModelType.MULTIMODAL,
    modalities=[ModalityType.TEXT, ModalityType.IMAGE],
    framework=["PyTorch", "Transformers"],
    vision_encoder_dim=1024,
)
```

### 4. Logging
```python
import logging

logger = logging.getLogger(__name__)

def load_model(self):
    logger.info(f"Loading model: {self.model_name_or_path}")
    # ... model loading code ...
    logger.info("Model loaded successfully")
```

Your model is now ready to be integrated into the KARMA evaluation framework! The system will automatically discover and make it available through the CLI and evaluation pipelines.
