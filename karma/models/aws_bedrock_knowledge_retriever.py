"""AWS Bedrock knowledge base retrieval model for KARMA.

This module provides a thin wrapper around the AWS Bedrock knowledge base
retrieval API so that queries originating from KARMA datasets can leverage the
managed hybrid search and reranking capabilities offered by Bedrock. The model
is designed to return the page numbers associated with each retrieved document,
which are needed for downstream retrieval metric computation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import boto3

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.data_models.model_meta import ModelMeta, ModelType, ModalityType
from karma.models.base_model_abs import BaseModel
from karma.registries.model_registry import register_model_meta


logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Container for a single retrieval result."""

    page_number: int
    source_url: Optional[str]
    score: Optional[float]
    metadata: Dict[str, Any]


class AWSBedrockKnowledgeRetriever(BaseModel):
    """Interact with AWS Bedrock knowledge bases for retrieval tasks."""

    def __init__(
        self,
        model_name_or_path: str = "aws-bedrock-knowledge-retriever",
        region_name: Optional[str] = None,
        knowledge_base_id: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        number_of_results: int = 3,
        override_search_type: str = "HYBRID",
        reranking_model_arn: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)

        self.model_type = ModelType.EMBEDDING
        self.modalities = [ModalityType.TEXT]

        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.knowledge_base_id = knowledge_base_id or os.getenv(
            "AWS_BEDROCK_KNOWLEDGE_BASE_ID"
        )
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )
        self.aws_session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN")
        self.number_of_results = number_of_results
        self.override_search_type = override_search_type
        self.reranking_model_arn = reranking_model_arn

        self.client = None
        self.load_model()

    def load_model(self) -> None:
        if not self.knowledge_base_id:
            raise ValueError(
                "knowledge_base_id must be provided either as an argument or via the "
                "AWS_BEDROCK_KNOWLEDGE_BASE_ID environment variable."
            )

        try:
            # Simple boto3 client like lucid - no Config overhead
            self.client = boto3.client(
                "bedrock-agent-runtime",
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
                region_name=self.region_name,
            )
            self.is_loaded = True
            logger.info(
                "Initialized AWS Bedrock knowledge base client for region %s",
                self.region_name,
            )
        except Exception as exc:
            logger.error(
                "Failed to initialize AWS Bedrock knowledge base client: %s", exc
            )
            raise

    def preprocess(
        self, inputs: List[DataLoaderIterable], **kwargs: Any
    ) -> List[Dict[str, Any]]:
        processed_inputs: List[Dict[str, Any]] = []
        for idx, item in enumerate(inputs):
            query = item.input
            if not query:
                logger.warning("Skipping empty query at index %d", idx)
                continue

            other_args = item.other_args or {}
            processed_inputs.append(
                {
                    "query": query,
                    "query_id": other_args.get("query_id", f"query-{idx}"),
                    "publisher": other_args.get("publisher"),
                }
            )

        return processed_inputs

    def _build_retrieval_configuration(self) -> Dict[str, Any]:
        configuration: Dict[str, Any] = {
            "vectorSearchConfiguration": {
                "overrideSearchType": self.override_search_type,
                "numberOfResults": self.number_of_results,
            }
        }

        if self.reranking_model_arn:
            configuration["vectorSearchConfiguration"]["rerankingConfiguration"] = {
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": {"modelArn": self.reranking_model_arn}
                },
            }

        return configuration

    def _retrieve_single(self, query: str) -> List[RetrievalResult]:
        try:
            response = self.client.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={"text": query},
                retrievalConfiguration=self._build_retrieval_configuration(),
            )
        except Exception as exc:  # pragma: no cover - network call
            logger.error("Failed to query AWS Bedrock knowledge base: %s ", exc)
            raise

        retrieval_results = response.get("retrievalResults", [])
        formatted_results: List[RetrievalResult] = []

        for result in retrieval_results:
            metadata = result.get("metadata", {})
            raw_page = metadata.get("x-amz-bedrock-kb-document-page-number")
            try:
                # AWS returns zero-indexed page numbers, so we normalise to one-indexed.
                page_number = int(raw_page) + 1 if raw_page is not None else -1
            except (TypeError, ValueError):
                page_number = -1

            formatted_results.append(
                RetrievalResult(
                    page_number=page_number,
                    source_url=metadata.get("source_url"),
                    score=result.get("score"),
                    metadata=metadata,
                )
            )

        return formatted_results

    def run(
        self, inputs: List[DataLoaderIterable], **kwargs: Any
    ) -> List[Dict[str, Any]]:
        if not self.is_loaded:
            raise RuntimeError("AWS Bedrock knowledge base client is not loaded.")

        processed = self.preprocess(inputs, **kwargs)
        if not processed:
            return []

        def _format_output(
            retrievals: List[RetrievalResult], query_id: str
        ) -> Dict[str, Any]:
            page_numbers = [
                result.page_number for result in retrievals if result.page_number > 0
            ]
            documents = [
                {
                    "page_number": result.page_number,
                    "source_url": result.source_url,
                    "score": result.score,
                    "metadata": result.metadata,
                }
                for result in retrievals
            ]

            return {
                "query_id": query_id,
                "page_numbers": page_numbers,
                "documents": documents,
            }

        # Sequential processing like lucid for best performance
        outputs: List[Dict[str, Any]] = []
        for item in processed:
            retrievals = self._retrieve_single(item["query"])
            outputs.append(_format_output(retrievals, item["query_id"]))

        return outputs

    def postprocess(self, model_outputs: Any, **kwargs: Any) -> Any:
        return model_outputs


aws_bedrock_knowledge_meta = ModelMeta(
    name="aws-bedrock-knowledge-retriever",
    description="AWS Bedrock knowledge base retrieval client that surfaces page numbers.",
    loader_class="karma.models.aws_bedrock_knowledge_retriever.AWSBedrockKnowledgeRetriever",
    loader_kwargs={},
    model_type=ModelType.EMBEDDING,
    modalities=[ModalityType.TEXT],
)

register_model_meta(aws_bedrock_knowledge_meta)
