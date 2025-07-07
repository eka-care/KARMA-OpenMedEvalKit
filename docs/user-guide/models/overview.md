# Models Guide

This comprehensive guide covers everything you need to know about working with models in KARMA - from using built-in models to creating sophisticated custom implementations for medical AI evaluation.

## Built-in Models

KARMA includes several pre-configured models optimized for medical AI evaluation across different modalities.

### Available Models Overview

```bash
# List all available models
karma list models

# Expected output:
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Model Name                                  ┃ Status      ┃ Modality           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ Qwen/Qwen3-0.6B                             │ ✓ Available │ Text               │
│ Qwen/Qwen3-1.7B                             │ ✓ Available │ Text               │
│ google/medgemma-4b-it                       │ ✓ Available │ Text               │
│ ai4bharat/indic-conformer-600m-multilingual │ ✓ Available │ Audio              │
│ aws-transcribe                              │ ✓ Available │ Audio              │
│ openai-whisper                              │ ✓ Available │ Audio              │
│ gemini-asr                                  │ ✓ Available │ Audio              │
└─────────────────────────────────────────────┴─────────────┴────────────────────┘
```

### Text Generation Models

#### Qwen Models
Alibaba's Qwen models with specialized thinking capabilities for medical reasoning:

```bash
# Get detailed model information
karma info model "Qwen/Qwen3-0.6B"

# Basic usage
karma eval --model "Qwen/Qwen3-0.6B" \
  --datasets openlifescienceai/pubmedqa

# Advanced configuration with thinking mode
karma eval --model "Qwen/Qwen3-0.6B" \
  --datasets openlifescienceai/pubmedqa \
  --model-kwargs '{"enable_thinking": true, "temperature": 0.3}'
```

**Key Features:**
- **Thinking Mode**: Explicit reasoning steps for complex medical problems
- **Multiple Sizes**: 0.6B, 1.7B parameters for different hardware requirements
- **Medical Optimization**: Fine-tuned performance on healthcare tasks
- **Multilingual**: Support for multiple languages including medical terminology

#### MedGemma Models
Google's medical-specialized Gemma models:

```bash
# MedGemma for specialized medical tasks
karma eval --model medgemma --model-path "google/medgemma-4b-it" \
  --datasets openlifescienceai/medmcqa \
  --model-kwargs '{"temperature": 0.1, "max_tokens": 512}'
```

**Key Features:**
- **Medical Specialization**: Pre-trained on medical literature and guidelines
- **Instruction Tuning**: Optimized for following medical instructions
- **Safety Features**: Built-in safety guardrails for medical applications
- **Research Grade**: Suitable for academic and clinical research

### Audio Recognition Models

#### IndicConformer ASR
AI4Bharat's Conformer model for Indian languages:

```bash
# Indian language speech recognition
karma eval \
  --model "ai4bharat/indic-conformer-600m-multilingual" \
  --model-path "ai4bharat/indic-conformer-600m-multilingual" \
  --datasets "ai4bharat/indicvoices_r" \
  --batch-size 1 \
  --dataset-args "ai4bharat/indicvoices_r:language=Hindi" \
  --processor-args "ai4bharat/indicvoices_r.general_text_processor:language=Hindi"
```

**Key Features:**
- **22 Indian Languages**: Complete coverage of constitutional languages
- **Medical Audio**: Optimized for healthcare speech recognition
- **Conformer Architecture**: State-of-the-art speech recognition architecture
- **Regional Dialects**: Handles diverse Indian language variations

#### Cloud ASR Services
Enterprise-grade speech recognition for production deployments:

```bash
# AWS Transcribe
karma eval --model aws-transcribe \
  --datasets "ai4bharat/indicvoices_r" \
  --model-kwargs '{"language_code": "en-US", "medical_vocabulary": true}'

# Google Cloud Speech
karma eval --model gemini-asr \
  --datasets "ai4bharat/indicvoices_r" \
  --model-kwargs '{"language": "en-US", "model": "medical_dictation"}'

# OpenAI Whisper
karma eval --model openai-whisper \
  --datasets "ai4bharat/indicvoices_r" \
  --model-kwargs '{"model": "whisper-1", "language": "en"}'
```

## Model Configuration and Optimization

### Parameter Tuning

#### Generation Parameters
Control model behavior with precision:

```bash
# Conservative generation for medical accuracy
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets openlifescienceai/pubmedqa \
  --model-kwargs '{
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 50,
    "max_tokens": 512,
    "enable_thinking": true,
    "seed": 42
  }'

# Creative generation for medical education
karma eval --model qwen --model-path "Qwen/Qwen3-0.6B" \
  --datasets medical_education_dataset \
  --model-kwargs '{
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 1024,
    "enable_thinking": false
  }'
```

#### Parameter Reference

| Parameter | Range | Description | Medical Use Case |
|-----------|-------|-------------|------------------|
| `temperature` | 0.0-1.0 | Randomness control | 0.1-0.3 for diagnostic accuracy |
| `top_p` | 0.0-1.0 | Nucleus sampling | 0.9 for balanced responses |
| `top_k` | 1-100 | Top-k sampling | 50 for medical terminology |
| `max_tokens` | 1-4096 | Output length | 512 for concise answers |
| `enable_thinking` | boolean | Reasoning mode | true for complex cases |
| `seed` | integer | Reproducibility | Set for consistent results |


## Custom Model Development

### Creating Text Generation Models

#### Basic Custom Model

```python
# karma/models/custom_medical_llm.py
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from karma.models.base_model_abs import BaseModel
from karma.data_models.model_meta import ModelMeta, ModelType, ModalityType
from karma.registries.model_registry import register_model_meta

class CustomMedicalLLM(BaseModel):
    """Custom medical language model with specialized preprocessing."""
    
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        temperature: float = 0.7,
        max_tokens: int = 512,
        medical_prompt_template: str = None,
        **kwargs
    ):
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.medical_prompt_template = medical_prompt_template or self._default_medical_template()
        
    def _default_medical_template(self) -> str:
        """Default medical prompt template."""
        return """You are a medical AI assistant. Please provide accurate, evidence-based responses.

Patient Case/Question: {question}

Instructions:
1. Analyze the medical information carefully
2. Provide a clear, concise answer
3. Include relevant medical reasoning
4. Cite evidence when applicable

Response:"""

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.is_loaded = True
        
    def preprocess(self, prompt: str, **kwargs) -> str:
        """Apply medical-specific preprocessing."""
        # Format with medical template
        formatted_prompt = self.medical_prompt_template.format(question=prompt)
        
        # Add medical context if provided
        medical_context = kwargs.get("medical_context", "")
        if medical_context:
            formatted_prompt = f"Medical Context: {medical_context}\n\n{formatted_prompt}"
            
        return formatted_prompt

    def run(self, inputs: List[str], **kwargs) -> List[str]:
        """Generate responses for medical queries."""
        if not self.is_loaded:
            self.load_model()
            
        responses = []
        
        for prompt in inputs:
            # Tokenize input
            formatted_prompt = self.preprocess(prompt, **kwargs)
            inputs_tokenized = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs_tokenized,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs_tokenized.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            responses.append(self.postprocess(generated_text, **kwargs))
            
        return responses
    
    def postprocess(self, response: str, **kwargs) -> str:
        """Apply medical-specific postprocessing."""
        # Clean up response
        response = response.strip()
        
        # Remove common artifacts
        response = response.replace("Response:", "").strip()
        
        # Apply medical formatting
        if kwargs.get("format_medical_response", True):
            response = self._format_medical_response(response)
            
        return response
    
    def _format_medical_response(self, response: str) -> str:
        """Format response for medical consistency."""
        # Ensure proper medical terminology capitalization
        medical_terms = {
            "covid-19": "COVID-19",
            "hiv": "HIV",
            "aids": "AIDS",
            "ecg": "ECG",
            "mri": "MRI",
            "ct scan": "CT scan"
        }
        
        for term, formatted_term in medical_terms.items():
            response = response.replace(term, formatted_term)
            response = response.replace(term.upper(), formatted_term)
            
        return response

# Model metadata
CUSTOM_MEDICAL_LLM_META = ModelMeta(
    name="custom_medical_llm",
    description="Custom medical language model with specialized medical processing",
    loader_class="karma.models.custom_medical_llm.CustomMedicalLLM",
    loader_kwargs={
        "temperature": 0.3,
        "max_tokens": 512,
        "device": "cuda"
    },
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
    framework=["PyTorch", "Transformers"],
    license="Custom",
    medical_specialization=True,
    supported_tasks=["medical_qa", "diagnosis_support", "medical_education"]
)

# Register the model
register_model_meta(CUSTOM_MEDICAL_LLM_META)
```

#### Using the Custom Model

```bash
# Use the custom medical model
karma eval --model custom_medical_llm \
  --model-path "microsoft/DialoGPT-medium" \
  --datasets openlifescienceai/pubmedqa \
  --model-kwargs '{
    "temperature": 0.2,
    "max_tokens": 256,
    "format_medical_response": true
  }'
```

### Creating Audio Recognition Models

#### Custom ASR Model

```python
# karma/models/custom_medical_asr.py
from typing import List, Dict, Any, Optional
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from karma.models.base_model_abs import BaseModel
from karma.data_models.dataloader_iterable import DataLoaderIterable

class CustomMedicalASR(BaseModel):
    """Custom ASR model specialized for medical audio."""
    
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        sample_rate: int = 16000,
        medical_vocabulary: List[str] = None,
        **kwargs
    ):
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)
        self.device = device
        self.sample_rate = sample_rate
        self.medical_vocabulary = medical_vocabulary or self._default_medical_vocab()
        
    def _default_medical_vocab(self) -> List[str]:
        """Default medical vocabulary for better recognition."""
        return [
            "stethoscope", "blood pressure", "heart rate", "temperature",
            "diagnosis", "symptom", "medication", "prescription",
            "hypertension", "diabetes", "pneumonia", "bronchitis"
        ]
        
    def load_model(self) -> None:
        """Load Wav2Vec2 model and processor."""
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name_or_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name_or_path)
        self.model.to(self.device)
        self.is_loaded = True
        
    def preprocess(self, input_item: DataLoaderIterable, **kwargs) -> torch.Tensor:
        """Preprocess audio for medical ASR."""
        # Load audio from bytes
        audio_bytes = input_item.audio
        audio_array, _ = librosa.load(
            io.BytesIO(audio_bytes),
            sr=self.sample_rate,
            mono=True
        )
        
        # Apply medical-specific audio preprocessing
        audio_array = self._enhance_medical_audio(audio_array)
        
        # Process with Wav2Vec2 processor
        inputs = self.processor(
            audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        return inputs.input_values.to(self.device)
    
    def _enhance_medical_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply medical-specific audio enhancements."""
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Apply band-pass filter for speech frequencies
        from scipy import signal
        sos = signal.butter(10, [300, 3400], btype='band', fs=self.sample_rate, output='sos')
        audio = signal.sosfilt(sos, audio)
        
        # Noise reduction for medical environments
        audio = self._reduce_medical_noise(audio)
        
        return audio
    
    def _reduce_medical_noise(self, audio: np.ndarray) -> np.ndarray:
        """Reduce common medical environment noise."""
        # Simple noise gate
        noise_threshold = 0.01
        audio[np.abs(audio) < noise_threshold] = 0
        
        return audio
        
    def run(self, inputs: List[DataLoaderIterable], **kwargs) -> List[str]:
        """Transcribe medical audio."""
        if not self.is_loaded:
            self.load_model()
            
        transcriptions = []
        
        for input_item in inputs:
            # Preprocess audio
            audio_tensor = self.preprocess(input_item, **kwargs)
            
            # Generate transcription
            with torch.no_grad():
                logits = self.model(audio_tensor).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.decode(predicted_ids[0])
                
            # Apply medical postprocessing
            transcription = self.postprocess(transcription, **kwargs)
            transcriptions.append(transcription)
            
        return transcriptions
    
    def postprocess(self, transcription: str, **kwargs) -> str:
        """Apply medical-specific postprocessing."""
        # Convert to lowercase for consistency
        transcription = transcription.lower()
        
        # Apply medical vocabulary corrections
        transcription = self._apply_medical_corrections(transcription)
        
        # Format medical terms
        transcription = self._format_medical_terms(transcription)
        
        return transcription.strip()
    
    def _apply_medical_corrections(self, text: str) -> str:
        """Apply medical vocabulary corrections."""
        corrections = {
            "blood presure": "blood pressure",
            "hart rate": "heart rate",
            "temperture": "temperature",
            "medcation": "medication",
            "diagosis": "diagnosis"
        }
        
        for error, correction in corrections.items():
            text = text.replace(error, correction)
            
        return text
    
    def _format_medical_terms(self, text: str) -> str:
        """Format medical terms consistently."""
        # Capitalize important medical terms
        medical_terms = ["COVID-19", "HIV", "AIDS", "ECG", "MRI", "CT"]
        
        for term in medical_terms:
            text = text.replace(term.lower(), term)
            
        return text
```

### Multimodal Models

#### Vision-Language Model for Medical Images

```python
# karma/models/medical_vision_language.py
from typing import List, Dict, Any, Optional, Union
import torch
from PIL import Image
import io
from transformers import BlipProcessor, BlipForConditionalGeneration
from karma.models.base_model_abs import BaseModel
from karma.data_models.dataloader_iterable import DataLoaderIterable

class MedicalVisionLanguageModel(BaseModel):
    """Vision-language model for medical image analysis."""
    
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        max_tokens: int = 512,
        medical_image_context: str = None,
        **kwargs
    ):
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)
        self.device = device
        self.max_tokens = max_tokens
        self.medical_image_context = medical_image_context or self._default_medical_context()
        
    def _default_medical_context(self) -> str:
        """Default medical imaging context."""
        return """Analyze this medical image carefully. Consider:
1. Anatomical structures visible
2. Any abnormalities or pathologies
3. Image quality and artifacts
4. Clinical significance

Provide a detailed, accurate medical description."""

    def load_model(self) -> None:
        """Load vision-language model."""
        self.processor = BlipProcessor.from_pretrained(self.model_name_or_path)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name_or_path)
        self.model.to(self.device)
        self.is_loaded = True
        
    def preprocess(self, input_item: DataLoaderIterable, **kwargs) -> Dict[str, torch.Tensor]:
        """Preprocess image and text for medical VL model."""
        # Load image
        if hasattr(input_item, 'image') and input_item.image:
            image = Image.open(io.BytesIO(input_item.image)).convert('RGB')
        else:
            raise ValueError("No image data provided")
            
        # Prepare text prompt
        text_prompt = kwargs.get('prompt', '') or getattr(input_item, 'text', '')
        if not text_prompt:
            text_prompt = "Describe this medical image in detail."
            
        # Add medical context
        full_prompt = f"{self.medical_image_context}\n\nQuery: {text_prompt}"
        
        # Process inputs
        inputs = self.processor(
            images=image,
            text=full_prompt,
            return_tensors="pt",
            padding=True
        )
        
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def run(self, inputs: List[DataLoaderIterable], **kwargs) -> List[str]:
        """Generate medical image descriptions."""
        if not self.is_loaded:
            self.load_model()
            
        descriptions = []
        
        for input_item in inputs:
            # Preprocess inputs
            model_inputs = self.preprocess(input_item, **kwargs)
            
            # Generate description
            with torch.no_grad():
                outputs = self.model.generate(
                    **model_inputs,
                    max_length=self.max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            # Decode and postprocess
            description = self.processor.decode(outputs[0], skip_special_tokens=True)
            description = self.postprocess(description, **kwargs)
            descriptions.append(description)
            
        return descriptions
    
    def postprocess(self, description: str, **kwargs) -> str:
        """Apply medical image analysis postprocessing."""
        # Remove prompt echo
        if "Query:" in description:
            description = description.split("Query:")[-1].strip()
            
        # Format medical terminology
        description = self._format_medical_terminology(description)
        
        # Add clinical structure if requested
        if kwargs.get('structured_output', False):
            description = self._structure_medical_report(description)
            
        return description
    
    def _format_medical_terminology(self, text: str) -> str:
        """Format medical terminology consistently."""
        # Medical acronyms
        acronyms = {
            "ct": "CT", "mri": "MRI", "x-ray": "X-ray",
            "ecg": "ECG", "ekg": "EKG", "pet": "PET"
        }
        
        for abbrev, formatted in acronyms.items():
            text = text.replace(f" {abbrev} ", f" {formatted} ")
            text = text.replace(f"{abbrev} ", f"{formatted} ")
            
        return text
    
    def _structure_medical_report(self, description: str) -> str:
        """Structure description as medical report."""
        return f"""
MEDICAL IMAGE ANALYSIS REPORT

FINDINGS:
{description}

IMPRESSION:
[Clinical interpretation based on findings]

RECOMMENDATIONS:
[Follow-up recommendations if applicable]
"""
```

### Integration with Other Components

Once you have mastered models, explore these interconnected topics:

- **[Datasets](../datasets/overview.md)** - Learn how to pair models with appropriate datasets
- **[Metrics](../metrics/overview.md)** - Understand how to evaluate model performance
- **[Processors](../processors/overview.md)** - Apply text processing to improve model inputs/outputs
- **[Configuration](../configuration/environment-setup.md)** - Optimize your environment for different models

### Advanced Topics

- **[API Reference](../../api-reference/models.md)** - Complete technical documentation
- **[Contributing](../contributing.md)** - Help improve KARMA's model ecosystem
- **[Research Papers](../research.md)** - Latest developments in medical AI models

### Community and Support

- **[GitHub Discussions](https://github.com/eka-care/KARMA-OpenMedEvalKit/discussions)** - Community support
- **[Model Registry](https://github.com/eka-care/KARMA-OpenMedEvalKit/wiki/Model-Registry)** - Community model contributions
- **[Benchmarks](https://github.com/eka-care/KARMA-OpenMedEvalKit/wiki/Benchmarks)** - Performance comparisons