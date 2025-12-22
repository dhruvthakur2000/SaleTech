from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from saletech.core.session_state import SessionState

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class Session(BaseModel):
    session_id: str
    state: SessionState = SessionState.IDLE
    history: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
