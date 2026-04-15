from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


class SessionState(str, Enum):
    """Session state machine"""
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    PROCESSING = "processing"
    INTERUPTED = "interupted"
    TERMINATED = "terminated"

class AudioChunk(BaseModel):
    """Audio data chunk"""
    session_id: str
    data: bytes
    timestamp: float
    sample_rate: int = 16000
    channels: int = 1


class VADResult(BaseModel):
    """Voice Activity Detection result"""
    is_speech: bool
    confidence: float
    timestamp: float

class Utterance(BaseModel):
    """Complete user utterance"""
    session_id: str
    audio_data: bytes
    start_time: float
    end_time: float
    duration_ms: float
    
    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0

class MessageRole(str, Enum):
    """Conversation Message Role"""
    SYSTEM= "system"
    USER= "user"
    ASSISTANT= "assistant"

class ConversationMessage(BaseModel):
    """Single Conversation"""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str,Any]] = None

class TranscriptionResult(BaseModel):
    """ASR transcription result"""
    text: str
    confidence: float
    language: Optional[str] = None
    utterance_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    latency_ms: Optional[float] = None
    audio_duration_ms: Optional[float] = None

class SessionMetrics(BaseModel):
    """Per-session performance metrics"""
    session_id: str
    
    # Latency tracking
    vad_latencies_ms: List[float] = Field(default_factory=list)
    asr_latencies_ms: List[float] = Field(default_factory=list)
    llm_latencies_ms: List[float] = Field(default_factory=list)
    tts_latencies_ms: List[float] = Field(default_factory=list)
    
    # Counters
    total_utterances: int = 0
    total_responses: int = 0
    interruptions: int = 0
    errors: int = 0
    
    # Session info
    started_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)

