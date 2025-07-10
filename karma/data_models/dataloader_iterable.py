from pydantic import BaseModel, Field, model_serializer
from PIL import Image
from typing import Any, Dict, List, Optional, Union


class ConversationTurn(BaseModel):
    content: str = Field(description="The content of the conversation.")
    role: str = Field(description="The role of the conversation.")


class Conversation(BaseModel):
    conversation: List[ConversationTurn]


class DataLoaderIterable(BaseModel):
    input: Optional[str] = Field(
        default=None,
        description="Input prompt passed as a sample from the dataset iter",
    )
    images: Optional[Union[Image.Image, List[Image.Image], bytes]] = Field(
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
    conversation: Optional[List[Conversation]] = Field(
        default=None,
        description="Conversation prompt passed as a sample from the dataset iter",
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


