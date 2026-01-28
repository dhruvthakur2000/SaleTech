from pydantic import BaseModel, Field
from typing import Optional

class STTEvalResult(BaseModel):
    model_name: str = Field(..., description="Name of the STT model")
    model_version: Optional[str] = Field(None, description="Version of the model")
    audio_file: str = Field(..., description="Path or name of the audio file tested")
    reference_text: str = Field(..., description="Ground truth transcription")
    predicted_text: str = Field(..., description="Model's predicted transcription")
    wer: float = Field(..., description="Word Error Rate")
    cer: Optional[float] = Field(None, description="Character Error Rate")
    latency_ms: Optional[float] = Field(None, description="Inference latency in milliseconds")
    gpu_memory_mb: Optional[float] = Field(None, description="GPU memory used in MB")
    cpu_used: Optional[bool] = Field(None, description="Whether CPU was used (True/False)")
    notes: Optional[str] = Field(None, description="Additional notes or metadata")

class STTEvalBatchResult(BaseModel):
    results: list[STTEvalResult] = Field(..., description="List of evaluation results for a batch of audio files")