# karma/models/docassistchat.py
import os

import requests
import uuid
import json
import time
from typing import List, Dict, Any
from karma.models.base_model_abs import BaseModel
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.registries.model_registry import register_model_meta
from karma.data_models.model_meta import ModelMeta, ModelType, ModalityType
import logging

logger = logging.getLogger(__name__)


class DocAssistChatModel(BaseModel):
    """DocAssistChat API model for medical question answering."""

    def __init__(
        self,
        model_name_or_path: str,
        base_url: str,
        d_oid: str,
        d_hash: str,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: int = 1,
        **kwargs,
    ):
        super().__init__(model_name_or_path, **kwargs)

        # API configuration
        self.base_url = base_url
        self.d_oid = d_oid
        self.d_hash = d_hash
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Headers for the API request
        self.headers = {
            "Content-Type": "application/json",
            "jwt-payload": json.dumps({"oid": self.d_oid}),
        }

    def load_model(self):
        """Initialize the API client - no actual model loading needed."""
        self.is_loaded = True

    def _generate_session_id(self) -> str:
        """Generate a unique session ID for each API call."""
        return f"karma-eval-{uuid.uuid4().hex[:12]}"

    def _make_api_request(self, messages: List[Dict[str, str]], session_id: str) -> str:
        """Make API request with SSE streaming and retry logic."""

        # Build query parameters
        query_params = {
            "d_oid": self.d_oid,
            "d_hash": self.d_hash,
            "session_id": session_id,
        }

        payload = json.dumps({"messages": messages})

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Making API request (attempt {attempt + 1}/{self.max_retries})"
                )

                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    data=payload,
                    params=query_params,
                    stream=True,  # Enable streaming for SSE
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    # Handle SSE streaming response
                    return self._parse_sse_response(response)
                else:
                    logger.warning(
                        f"API request failed with status {response.status_code}: {response.text}"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(
                            self.retry_delay * (2**attempt)
                        )  # Exponential backoff
                        continue
                    else:
                        return f"API Error: {response.status_code} - {response.text}"

            except requests.exceptions.Timeout:
                logger.warning(
                    f"API request timed out (attempt {attempt + 1}/{self.max_retries})"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))
                    continue
                else:
                    return "API Error: Request timed out"

            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"API request failed: {e} (attempt {attempt + 1}/{self.max_retries})"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))
                    continue
                else:
                    return f"API Error: {str(e)}"

        return "API Error: Max retries exceeded"

    def _parse_sse_response(self, response: requests.Response) -> str:
        """Parse Server-Sent Events (SSE) streaming response."""
        try:
            last_response = None

            # Iterate over streaming lines
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")

                    # SSE format: lines starting with "data: "
                    if decoded_line.startswith("data: "):
                        last_response = decoded_line[6:]  # Remove 'data: ' prefix

            # Return the last received data chunk
            if last_response:
                try:
                    # Try to parse as JSON in case it's structured
                    json_response = json.loads(last_response)

                    # Extract text content from various possible fields
                    if isinstance(json_response, dict):
                        if "content" in json_response:
                            return json_response["content"]
                        elif "text" in json_response:
                            return json_response["text"]
                        elif "message" in json_response:
                            return json_response["message"]
                        elif "response" in json_response:
                            return json_response["response"]
                        else:
                            return str(json_response)
                    else:
                        return str(json_response)

                except json.JSONDecodeError:
                    # If not JSON, return as plain text
                    return last_response
            else:
                return "No response received from SSE stream"

        except Exception as e:
            logger.error(f"Error parsing SSE response: {e}")
            # Fallback to getting response text
            try:
                return response.text
            except:
                return f"Error parsing response: {str(e)}"

    def preprocess(self, inputs: List[DataLoaderIterable]) -> List[Dict[str, Any]]:
        """Convert KARMA inputs to DocAssistChat API format."""
        processed_inputs = []

        for item in inputs:
            # Generate unique session ID for each input
            session_id = self._generate_session_id()

            # Handle different input types
            if item.conversation:
                # Multi-turn conversation
                messages = []
                if item.system_prompt:
                    # Prepend system prompt to user message
                    messages.append({"role": "user", "text": item.system_prompt})
                for turn in item.conversation.conversation_turns:
                    messages.append({"role": turn.role, "text": turn.content})
            else:
                # Single input - create a user message
                text_input = item.input
                if item.system_prompt:
                    # Prepend system prompt to user message
                    text_input = f"System: {item.system_prompt}\n\nUser: {text_input}"

                messages = [{"role": "user", "text": text_input}]

            processed_inputs.append({"messages": messages, "session_id": session_id})
        return processed_inputs

    def run(self, inputs: List[DataLoaderIterable]) -> List[str]:
        """Generate model outputs using DocAssistChat API."""
        if not self.is_loaded:
            self.load_model()

        # Preprocess inputs
        processed_inputs = self.preprocess(inputs)

        responses = []

        for processed_input in processed_inputs:
            messages = processed_input["messages"]
            session_id = processed_input["session_id"]

            logger.debug(f"Processing request with session_id: {session_id}")

            # Make API request
            response = self._make_api_request(messages, session_id)
            responses.append(response)

        return self.postprocess(responses)

    def postprocess(self, outputs: List[str]) -> List[str]:
        """Clean up generated outputs."""
        cleaned_outputs = []

        for output in outputs:
            # Clean up the response
            cleaned = output.strip()

            # Remove any API error prefixes if they exist
            if cleaned.startswith("API Error:"):
                logger.warning(f"API error in response: {cleaned}")

            cleaned_outputs.append(cleaned)

        return cleaned_outputs


# Model metadata definitions
DocAssistChatDefault = ModelMeta(
    name="docassistchat/default",
    description="DocAssistChat API model for medical question answering with SSE streaming",
    loader_class="karma.models.docassist_chat.DocAssistChatModel",
    loader_kwargs={
        "base_url": "http://lucid.eka.care/doc_chat/v1/stream_chat",
        "d_oid": os.getenv("DOC_ASSIST_CHAT_D_OID"),
        "d_hash": os.getenv("DOC_ASSIST_CHAT_D_HASH"),
        "timeout": 30,
        "max_retries": 3,
        "retry_delay": 1,
    },
    model_type=ModelType.TEXT_GENERATION,
    modalities=[ModalityType.TEXT],
)

# Register the model
register_model_meta(DocAssistChatDefault)
