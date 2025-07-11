from pydantic import BaseModel, Field, model_serializer
from PIL import Image
from typing import Any, Dict, List, Optional, Union


class ConversationTurn(BaseModel):
    content: str = Field(description="The content of the conversation.")
    role: str = Field(description="The role of the conversation.")


class Conversation(BaseModel):
    conversation_turns: List[ConversationTurn]


class RubricCriteria(BaseModel):
    criterion: str = Field(description="The rubric criterion text.")
    points: float = Field(description="Points awarded if criterion is met.")
    tags: List[str] = Field(
        default_factory=list, description="Tags for categorizing rubric items."
    )


class DataLoaderIterable(BaseModel):
    input: Optional[str] = Field(
        default=None,
        description="Input prompt passed as a sample from the dataset iter",
    )
    images: Optional[Union[Image.Image, List[Image.Image], bytes, List[bytes]]] = Field(
        default=None,
        description="Image prompt passed as a sample from the dataset iter. See medxpertqa",
    )
    expected_output: Optional[Any] = Field(
        default=None,
    )
    audio: Optional[Any] = Field(
        default=None,
        description="Audio prompt passed as a sample from the dataset iter",
    )
    other_args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Other arguments passed as a sample from the dataset iter",
    )
    conversation: Optional[Conversation] = Field(
        default=None,
        description="Conversation prompt passed as a sample from the dataset iter",
    )
    rubric_to_evaluate: Optional[List[RubricCriteria]] = Field(
        default=None, description="As defined in the healthbench paper, provide rubric"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="The system prompt in case applicable passed as a sample from the dataset iter",
    )

    class Config:
        arbitrary_types_allowed = True
        exclude_none = True
        exclude_unset = True
        exclude_defaults = True

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        # Custom serialization logic that excludes None values
        return {k: v for k, v in self.__dict__.items() if v is not None}
