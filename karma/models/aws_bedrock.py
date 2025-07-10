import os
from typing import List

import boto3

from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.models.base_model_abs import BaseModel


class AWSBedrock(BaseModel):
    def __init__(
        self,
        model_name_or_path: str = "anthropic.claude-3-5-sonnet-20240620-v1%3A0",
        region_name: str = os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        **kwargs,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            **kwargs,
        )
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.client = None
        self.load_model()

    def load_model(self):
        self.client = boto3.client("bedrock-runtime")
        self.is_loaded = True

    def run(self, inputs: List[DataLoaderIterable], **kwargs):
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")
        outputs = []
        for i in inputs:
            response = self.model.converse(
                modelId=self.model_name_or_path,
                system=[
                    {
                        "text": i.input,
                    }
                ],
            )
            out = response["output"]["message"]["content"]["text"]
            outputs.append(out)
        return outputs
