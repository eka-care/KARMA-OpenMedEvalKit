from karma.models.base_model_abs import BaseModel
from karma.data_models.dataloader_iterable import DataLoaderIterable
from openai import OpenAI
from typing import List
import os

class OpenAIClinet(BaseModel):
    def __init__(self,
                 model_name_or_path: str = "gpt-4o",
                 api_key: str = os.getenv("OPENAI_API_KEY"),
                 **kwargs):
        super().__init__(
            model_name_or_path=model_name_or_path,
            **kwargs,
        )
        self.api_key = api_key
        self.client = None
        self.load_model()

    def load(self):
        self.client = OpenAI(api_key=self.api_key)
        self.is_loaded = True

    def run(self, inputs: List[DataLoaderIterable], **kwargs):
        response = self.client.responses.create(
            model = self.model_name_or_path,
            input =  [{"role": "user", "content": RUBRICS_PROMPT}], #Prompt
            **kwargs
        )
        return response.output_text