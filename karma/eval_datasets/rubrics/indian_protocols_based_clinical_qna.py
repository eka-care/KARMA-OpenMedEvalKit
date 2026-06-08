"""
Indian Protocols-Based Clinical Q&A Conversation Evaluation Dataset.

Rubric-based evaluation dataset for Indian medical protocol conversations.
Each sample carries two rubric types:
- query_rubrics (stage:retrieval_query) — evaluate the bot's tool/retrieval queries
- answer_rubrics (stage:final_answer) — evaluate the bot's final clinical answer

Loads from HuggingFace (ekacare/indian_protocols_based_clinical_QnA) by default,
or from a local Arrow dataset path if provided.

The single boolean ``has_tools`` flag (default False) is the source of truth for
whether the run is a tools run:
- has_tools=False -> uses BASE_SYSTEM_PROMPT, query_rubric_evaluation will short-circuit
- has_tools=True  -> uses TOOL_AUGMENTED_SYSTEM_PROMPT, query_rubric_evaluation runs

Usage:
    karma eval --model us.anthropic.claude-sonnet-4-6 \\
        --datasets "ekacare/indian_protocols_based_clinical_QnA" \\
        --dataset-args has_tools=true \\
        --model-args '{"tools": ["http://localhost:8000/mcp"], "tool_trace": true}' \\
        --metric-args "answer_rubric_evaluation:provider_to_use=openai,model_id=gpt-5.1,max_workers=20" \\
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

DATASET_NAME = "ekacare/indian_protocols_based_clinical_QnA"

BASE_SYSTEM_PROMPT = """\
You are a medical assistant with deep knowledge of clinical guidelines, \
Indian medical protocols, and pharmacology. You help experienced doctors \
by answering clinical queries concisely and accurately.

RESPONSE RULES:
- Provide direct, evidence-based answers. Be concise but clinically complete.
- Use markdown formatting: headers (h2/h3), bullet points, and tables where helpful.
- When the doctor's query references a specific guideline or publisher, \
answer from that guideline's recommendations based on your training knowledge.

CRITICAL — NEVER REFUSE OR DEFER:
- This is a single-turn interaction. There is no follow-up.
- NEVER ask the user to clarify, choose between options, or confirm anything.
- NEVER say "please select", "which one do you mean", or "reply with 1/2/3".
- ALWAYS provide your best answer in a single response, even if you are uncertain.
- If a query is ambiguous, state your interpretation briefly and answer it.
"""

TOOL_AUGMENTED_SYSTEM_PROMPT = """\
You are a medical assistant with deep knowledge of clinical guidelines, \
Indian medical protocols, and pharmacology. You help experienced doctors \
by answering clinical queries concisely and accurately.

RESPONSE RULES:
- Provide direct, evidence-based answers. Be concise but clinically complete.
- Use markdown formatting: headers (h2/h3), bullet points, and tables where helpful.
- When the doctor's query references a specific guideline or publisher, \
answer from that guideline's recommendations.

TOOL USAGE:
- The medai-tools MCP server exposes `indian_treatment_protocol_search` for \
searching published Indian and international medical protocols. Use it whenever \
the doctor's query references a guideline, publisher, treatment protocol, \
diagnostic criteria, drug selection per protocol, or threshold values.
- Workflow: call `indian_treatment_protocol_search` with intent="publishers" \
first if the publisher is ambiguous; then call again with intent="search" and \
a list of {query, publisher} objects. Keep queries concise — use clinical \
keywords, not question words. Break broad questions into multiple targeted \
sub-queries.
- When the query references a publisher acronym, map it to the closest match:
  - "JH" likely refers to "Journal of Hypertension" (European Society of Hypertension)
  - "STG 2022" under an Indian pediatrics context likely refers to IAP or MoHFW
  - "RSSDI" refers to the Research Society for the Study of Diabetes in India
  - "ACG" refers to the American College of Gastroenterology
- The medai-tools MCP also exposes `indian_branded_drug_search`, \
`indian_pharmacology_details`, and a `medical_calculator_*` family — use them \
when the query is about Indian-branded medicines, NFI 2011 generic details, or \
medical calculations respectively.
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
    metrics=["answer_rubric_evaluation", "query_rubric_evaluation"],
    task_type="conversation_rubric_evaluation",
    optional_args=["dataset_path", "system_prompt", "has_tools"],
)
class IndianProtocolsClinicalQnA(BaseMultimodalDataset):
    """
    Indian Protocols-Based Clinical Q&A Conversation Dataset with dual rubric evaluation.

    Each sample contains query_rubrics (stage:retrieval_query) for evaluating
    retrieval queries and answer_rubrics (stage:final_answer) for evaluating
    clinical answers.

    Loads from HuggingFace by default. Pass dataset_path to load from a local
    Arrow dataset instead.
    """

    def __init__(
        self,
        dataset_path: str = None,
        system_prompt: str = None,
        has_tools: bool = False,
        **kwargs,
    ):
        self.dataset_name = DATASET_NAME

        if isinstance(has_tools, str):
            has_tools = has_tools.lower() == "true"
        self.has_tools = bool(has_tools)

        if system_prompt is not None:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = (
                TOOL_AUGMENTED_SYSTEM_PROMPT if self.has_tools else BASE_SYSTEM_PROMPT
            )

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
            "Indian protocols Q&A dataset loaded with %d samples (has_tools=%s)",
            len(self.ds),
            self.has_tools,
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

        doctor_query = sample["doctor_query"]

        conversation = Conversation(
            conversation_turns=[ConversationTurn(content=doctor_query, role="user")]
        )

        doctor_persona = sample.get("doctor_persona", "{}")
        if isinstance(doctor_persona, str):
            doctor_persona = json.loads(doctor_persona)

        other_args = {
            "prompt_id": prompt_id,
            "has_tools": self.has_tools,
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
        }

        return DataLoaderIterable(
            conversation=conversation,
            rubric_to_evaluate=criterions,
            system_prompt=self.system_prompt,
            other_args=other_args,
        )

    def extract_prediction(self, response: str, **kwargs):
        return response, True
