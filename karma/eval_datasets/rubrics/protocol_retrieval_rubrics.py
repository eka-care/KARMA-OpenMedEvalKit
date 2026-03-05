"""
Protocol Retrieval Rubrics Conversation Evaluation Dataset.

Rubric-based evaluation dataset for Indian medical protocol conversations.
Each sample has two rubric types:
- query_rubrics (stage:retrieval_query) — evaluates the bot's retrieval query
- answer_rubrics (stage:final_answer) — evaluates the bot's clinical answer

Loads from HuggingFace (ekacare/protocol_retrieval_rubrics) by default, or
from a local Arrow dataset path if provided.

Usage:
    karma eval --model us.anthropic.claude-sonnet-4-6 \
        --datasets "ekacare/protocol_retrieval_rubrics" \
        --metric-args "answer_rubric_evaluation:provider_to_use=openai,model_id=gpt-5-mini,max_workers=20" \
        --batch-size 20
"""

import json
import logging
from typing import Any, Dict, List

import datasets

from karma.data_models.dataloader_iterable import (
    Conversation,
    ConversationTurn,
    DataLoaderIterable,
    RubricCriteria,
)
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset

logger = logging.getLogger(__name__)

DATASET_NAME = "ekacare/protocol_retrieval_rubrics"

OPENMED_SYSTEM_PROMPT = """\
You are a medical assistant with deep knowledge of clinical guidelines, \
Indian medical protocols, and pharmacology. You help experienced doctors \
by answering clinical queries concisely and accurately.

RESPONSE RULES:
- Provide direct, evidence-based answers. Be concise but clinically complete.
- Use markdown formatting: headers (h2/h3), bullet points, and tables where helpful.
- When the doctor's query references a specific guideline or publisher, \
answer from that guideline's recommendations.

TOOL USAGE:
- If tools are available, use them to retrieve protocol content relevant to the query.
- When searching for protocols, pick the publisher that best matches the \
doctor's query. If the exact publisher name is ambiguous, select the closest \
match based on your medical knowledge. For example:
  - "JH" likely refers to "Journal of Hypertension" (European Society of Hypertension)
  - "STG 2022" under an Indian pediatrics context likely refers to IAP or MoHFW
  - "RSSDI" refers to the Research Society for the Study of Diabetes in India
- If tool results are empty or irrelevant, answer from your own medical knowledge.
- Treat tool results as supplementary context — if they conflict with \
well-established guidelines, prefer the established evidence.

CRITICAL — NEVER REFUSE OR DEFER:
- This is a single-turn interaction. There is no follow-up.
- NEVER ask the user to clarify, choose between options, or confirm anything.
- NEVER say "please select", "which one do you mean", or "reply with 1/2/3".
- ALWAYS provide your best answer in a single response, even if you are uncertain.
- If a query is ambiguous, state your interpretation briefly and answer it.
"""


@register_dataset(
    dataset_name=DATASET_NAME,
    split="test",
    metrics=["answer_rubric_evaluation"],
    task_type="conversation_rubric_evaluation",
    optional_args=["dataset_path", "system_prompt"],
)
class ProtocolRetrievalRubrics(BaseMultimodalDataset):
    """
    Protocol Retrieval Rubrics Conversation Dataset with dual rubric evaluation.

    Each sample contains query_rubrics (stage:retrieval_query) for evaluating
    retrieval queries and answer_rubrics (stage:final_answer) for evaluating
    clinical answers.

    Loads from HuggingFace by default. Pass dataset_path to load from a local
    Arrow dataset instead.
    """

    def __init__(
        self,
        dataset_path: str = None,
        system_prompt: str = OPENMED_SYSTEM_PROMPT,
        **kwargs,
    ):
        self.dataset_name = DATASET_NAME
        self.system_prompt = system_prompt
        max_samples_raw = kwargs.get("max_samples", None)
        self.max_samples = int(max_samples_raw) if max_samples_raw is not None else None
        self.processors = kwargs.get("processors", [])
        self.split = "test"

        if dataset_path:
            ds_dict = datasets.load_from_disk(dataset_path)
        else:
            ds_dict = datasets.load_dataset(DATASET_NAME)

        if isinstance(ds_dict, datasets.DatasetDict):
            self.ds = ds_dict["test"]
        else:
            self.ds = ds_dict

        if self.max_samples:
            self.ds = self.ds.select(range(min(self.max_samples, len(self.ds))))

        logger.info(
            "OpenMed Conversation dataset loaded with %d samples from %s",
            len(self.ds),
            dataset_path,
        )

    def __iter__(self):
        for i in range(len(self.ds)):
            yield self.format_item(self.ds[i])

    def __len__(self):
        return len(self.ds)

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        prompt_id = sample.get("prompt_id", "")

        # Parse rubrics
        query_rubrics_raw = sample.get("query_rubrics", "[]")
        answer_rubrics_raw = sample.get("answer_rubrics", "[]")

        if isinstance(query_rubrics_raw, str):
            query_rubrics_raw = json.loads(query_rubrics_raw)
        if isinstance(answer_rubrics_raw, str):
            answer_rubrics_raw = json.loads(answer_rubrics_raw)

        criterions: List[RubricCriteria] = []
        for rubric in query_rubrics_raw:
            criterions.append(
                RubricCriteria(
                    criterion=rubric.get("criterion", ""),
                    points=rubric.get("points", 1),
                    tags=rubric.get("tags", []),
                )
            )
        for rubric in answer_rubrics_raw:
            criterions.append(
                RubricCriteria(
                    criterion=rubric.get("criterion", ""),
                    points=rubric.get("points", 1),
                    tags=rubric.get("tags", []),
                )
            )

        # Build single-turn conversation from doctor_query
        doctor_query = sample.get("doctor_query", "")
        if not doctor_query:
            # Fallback to prompt field
            prompt_data = sample.get("prompt", "[]")
            if isinstance(prompt_data, str):
                prompt_data = json.loads(prompt_data)
            if prompt_data and isinstance(prompt_data, list):
                doctor_query = prompt_data[0].get("content", "")

        conversation = Conversation(
            conversation_turns=[ConversationTurn(content=doctor_query, role="user")]
        )

        # Parse example_tags
        example_tags = sample.get("example_tags", "[]")
        if isinstance(example_tags, str):
            example_tags = json.loads(example_tags)

        # Parse doctor_persona
        doctor_persona = sample.get("doctor_persona", "{}")
        if isinstance(doctor_persona, str):
            doctor_persona = json.loads(doctor_persona)

        other_args = {
            "prompt_id": prompt_id,
            "query_style": sample.get("query_style", ""),
            "question_type": sample.get("question_type", ""),
            "difficulty_level": sample.get("difficulty_level", 1),
            "publisher": sample.get("publisher", ""),
            "doctor_persona": doctor_persona,
            "patient_case": sample.get("patient_case", ""),
            "scenario": sample.get("scenario", ""),
            "ground_truth_query": sample.get("ground_truth_query", ""),
            "answer_citation": sample.get("answer_citation", ""),
            "corpus_id": sample.get("corpus_id", ""),
            "example_tags": example_tags,
        }

        return DataLoaderIterable(
            conversation=conversation,
            rubric_to_evaluate=criterions,
            system_prompt=self.system_prompt,
            other_args=other_args,
        )

    def extract_prediction(self, response: str, **kwargs):
        return response, True
