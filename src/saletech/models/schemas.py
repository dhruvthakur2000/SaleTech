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

