from enum import Enum

class SessionState(str,Enum):
    IDLE='idle'
    LISTENING='listening'
    THINKING='thinking'
    SPEAKING='speaking'
    CLOSED='closed' 