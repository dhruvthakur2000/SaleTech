from enum import Enum

class SessionState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    PROCESSING = "processing"
    TERMINATED = "terminated"
