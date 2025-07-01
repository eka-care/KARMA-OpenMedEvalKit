from pydantic import BaseModel, Field
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class DataLoaderIterable(BaseModel):
    input: Optional[str] = Field(
        default=None,
        description="Input prompt passed as a sample from the dataset iter",
    )
    images: Optional[Union[Image, List[Image]]] = Field(
        default=None,
        description="Image prompt passed as a sample from the dataset iter. See medxpertqa",
    )
    expected_output: Optional[Union[str]] = Field(
        default=None,
    )
    audio: Optional[Any] = Field(
        default=None,
        description="Audio prompt passed as a sample from the dataset iter",
    )
