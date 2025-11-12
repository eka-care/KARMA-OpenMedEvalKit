import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Dict
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from huggingface_hub import login
from karma.registries.model_registry import register_model_meta
from karma.data_models.model_meta import ModelMeta, ModelType, ModalityType
from karma.models.base_model_abs import BaseModel
from karma.utils.retrieval.pooling import cls_pooling, mean_pooling, eos_pooling
from karma.utils.retrieval.index import faiss_index
from transformers import AutoTokenizer, AutoModel

hf_token = os.getenv("HF_TOKEN")
class HFEmbeddingBaseModel(BaseModel):
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int,
        device: str = "cuda",
        pooling: str = "mean",
        k_values: Optional[List[int]] = [1, 3, 5, 10],
        quantization: bool = False,
        score_function = "cos_sim", # ["cos_sim", "dot_product"]
        dataset_name: str = "ekacare/Eka-IndicMTEB",
        **kwargs,
    ):
        """
        Initialize Embedding model.

        Args:
            model_path: Path to the model (HuggingFace model ID)
            device: Device to use for inference ("auto", "cuda", "cpu")
            max_length: Maximum length of sequences to use for inference
            quantization: True for 4bit quantization; False for No quantization
            score_function: Similarity function to use in search
            **kwargs: Additional model-specific parameters
        """
        # Initialize parent class
        super().__init__(
            model_name_or_path=model_name_or_path,
            max_length=max_length,
            device=device,
            pooling=pooling,
            quantization=quantization,
            score_function=score_function,
            k_values=k_values,
            dataset_name=dataset_name,
            **kwargs,
        )
        # Model arguments
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.quantization = quantization
        self.device = device
        self.score_function = score_function
        self.logger = logging.getLogger(__name__)
        
        self.k_values = k_values
        self.max_retrieval_k = max(self.k_values)

        if self.score_function not in ["cos_sim", "dot_product"]:
            raise ValueError(f"Score function must be one of {['cos_sim', 'dot_product']}.")

        pooling_functions = {"cls": cls_pooling, "mean": mean_pooling, "eos": eos_pooling}
        if pooling not in pooling_functions.keys():
            raise ValueError("pooling must be one of {}".format(pooling_functions.keys()))

        self.logger.info(f"Using {pooling} pooling")
        self.pooling_func = pooling_functions[pooling]

        self.dataset_name = dataset_name
        self.corpus = load_dataset(self.dataset_name, "corpus", token=os.getenv("HF_TOKEN"))["test"]
        self.logger.info(f"Corpus length: {len(self.corpus)}")

        self.faiss_wrapper = faiss_index.FAISSIndex(score_function=self.score_function, logger=self.logger)
        self._build_index()
    
    def _build_index(self):
        self.load_model()
    
        texts = [row.get("text") for row in self.corpus]
        embs = self._batch_encode_corpus(texts, batch_size=128)

        # Create index and add vectors
        self.faiss_wrapper.create_index(dimension=embs.shape[1])
        self.faiss_wrapper.add(embs)

    def load_model(self):
        """Load main embedding model"""
        quant_config = None
        if self.quantization:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.logger.info(f"Using 4 bit quantization config: {quant_config}")

        self.processor = AutoTokenizer.from_pretrained(self.model_name_or_path)

        self.model = AutoModel.from_pretrained(
            self.model_name_or_path,
            device_map="auto",
            quantization_config=quant_config
        )

    def _batch_encode_corpus(self, texts, batch_size=32):
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Building Index"):
            batch = texts[i : i + batch_size]
            batch_embs = self.get_embeddings(batch)
            all_embeddings.append(batch_embs)

        return np.vstack(all_embeddings)

    def get_embeddings(self, queries):
        # Get embeddings for queries
        model_inputs = self.processor(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**model_inputs)
            embeddings = self.pooling_func(
                outputs, attention_mask=model_inputs["attention_mask"]
            )

        embeddings = np.ascontiguousarray(embeddings.cpu().numpy(), dtype=np.float32)
        return embeddings

    def run(self, inputs):
        # Prepare inputs for tokenizer (list of queries)
        query_ids = [item.other_args["query_id"] for item in inputs]
        queries = [item.input for item in inputs]

        # get embeddings
        embeddings = self.get_embeddings(queries)
        retrieval_results = self.faiss_wrapper.retrieve(embeddings, query_ids, self.corpus, self.max_retrieval_k)
        
        return retrieval_results
            
    
    def preprocess(self, inputs):
        return inputs

    def postprocess(self, outputs):        
        return outputs


model_paths = [
    "ekacare/parrotlet-e",
    "ekacare/SapBERT-finetuned",
    "ekacare/EmbeddingGemma-finetuned",
    "ekacare/IndicBertV2-finetuned",
]
for model_path in model_paths:
    model = ModelMeta(
        name=model_path,
        loader_class="karma.models.hf_embedding_model.HFEmbeddingBaseModel",
        loader_kwargs={
            "model_name_or_path": model_path,
            "max_length": 60,
            "k_values": [1, 3, 5],
            "quantization": None,
        },
        model_type=ModelType.EMBEDDING,
        modalities=[ModalityType.TEXT]
    )
    register_model_meta(model)