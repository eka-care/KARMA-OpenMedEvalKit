import os
import time
import uuid
import logging
from typing import List
import boto3
import requests
from karma.models.base_model_abs import BaseModel
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.data_models.model_meta import ModelMeta, ModalityType, ModelType
from karma.registries.model_registry import register_model_meta
from karma.utils.retry import call_with_retry

logger = logging.getLogger(__name__)


class AWSTranscribeASR(BaseModel):
    """AWS Transcribe-based ASR model for the KARMA framework."""
    
    def __init__(
        self,
        model_name_or_path: str = "aws-transcribe",
        region_name: str = None,
        s3_bucket: str = None,
        s3_key_prefix: str = None,
        **kwargs
    ):
        """
        Initialize the AWS Transcribe ASR service.
        
        Args:
            region_name: AWS region for Transcribe service
            s3_bucket: S3 bucket name for temporary audio storage
            s3_key_prefix: S3 key prefix for temporary files
            **kwargs: Additional arguments passed to BaseModel
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            **kwargs,
        )
        
        self.region_name = region_name or os.getenv("AWS_REGION")
        self.s3_bucket = s3_bucket or os.getenv("AWS_S3_BUCKET")
        self.s3_key_prefix = s3_key_prefix or os.getenv("AWS_S3_KEY_PREFIX")
        self.client = None
        self.load_model()
    
    def load_model(self):
        """Initialize the AWS Transcribe client."""
        self.client = boto3.client('transcribe', region_name=self.region_name)
    

    def upload_obj_to_s3(self, bucket, key, body):
        s3_resource = boto3.resource('s3')
        s3_object = s3_resource.Object(bucket, key)
        s3_object.put(Body=body)

    def transcribe_audio(self, s3_url):
        job_name = f"transcription_{str(uuid.uuid1())}"
        response = self.client.start_transcription_job(
            TranscriptionJobName=job_name,
            IdentifyLanguage=True,
            MediaFormat='mp3',
            Media={'MediaFileUri': s3_url}
        )
        while True:
            status = self.client.get_transcription_job(TranscriptionJobName=job_name)
            if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                break
            time.sleep(0.10)

        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            response = requests.get(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])
            transcript = response.json()
            transcription_result = ""
            for result in transcript['results']['transcripts']:
                transcription_result += result['transcript'] + '\n'
            return transcription_result
        else:
            raise RuntimeError(f"Failed to transcribe with AWS Transcribe: {status['TranscriptionJob']['TranscriptionJobStatus']}")
        
    def transcribe(self, audio_bytes):
        def _call():
            s3_key = f"{self.s3_key_prefix}temp_{uuid.uuid4()}.mp4"
            self.upload_obj_to_s3(self.s3_bucket, s3_key, audio_bytes)
            return self.transcribe_audio(f"s3://{self.s3_bucket}/{s3_key}")

        return call_with_retry(_call, description="AWS Transcribe")
    
    def preprocess(self, inputs: List[DataLoaderIterable], **kwargs):
        """
        Preprocess audio inputs for transcription.
        
        Args:
            inputs: List of DataLoaderIterable objects containing audio data
            
        Returns:
            List of audio items ready for processing
        """
        audio_items = []
        for item in inputs:
            audio_items.append(item.audio)
        return audio_items
    
    def run(self, inputs: List[DataLoaderIterable], **kwargs):
        """
        Run transcription on the input audio files.

        Args:
            inputs: List of DataLoaderIterable objects containing audio data

        Returns:
            List of {"text", "error"} dicts, one per input
        """
        results = []
        audio_items = self.preprocess(inputs)
        for i, audio_item in enumerate(audio_items):
            try:
                results.append({"text": self.transcribe(audio_item), "error": None})
            except Exception as e:
                logger.warning("AWS Transcribe failed for input %d: %s", i, e)
                results.append({"text": "", "error": str(e)})
        return results

    def run_with_metadata(self, inputs: List[DataLoaderIterable], **kwargs):
        outputs = self.run(inputs, **kwargs)
        return [
            {
                "text": (out.get("text") or "").strip(),
                "tool_trace": "",
                "error": out.get("error"),
            }
            for out in outputs
        ]
    
    
    def postprocess(self, transcriptions: List[str], **kwargs):
        """
        Postprocess transcriptions (currently just returns them as-is).
        
        Args:
            transcriptions: List of transcribed text strings
            
        Returns:
            List of processed transcriptions
        """
        return transcriptions


# Model metadata definitions
AWSTranscribeASRModel = ModelMeta(
    name="aws-transcribe",
    description="AWS Transcribe ASR model with automatic language detection",
    loader_class="karma.models.aws_transcribe_asr.AWSTranscribeASR",
    loader_kwargs={},
    revision=None,
    reference="https://docs.aws.amazon.com/transcribe/",
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    release_date="2017-11-29",
    version="1.0",
)

# Register the model
register_model_meta(AWSTranscribeASRModel) 