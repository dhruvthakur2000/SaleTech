from src.saletech.utils.logger import get_logger
from typing import Optional

class SaleTechException(Exception):
    """BAse exception for SAletech Errors."""
    def __init__(self,
                 message: str,
                 error_code: str,
                 status_code: int = 500,
                 context: dict = None,
                 original_exception: Exception = None):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.context = context or {}
        self.original_exception = original_exception
        super().__init__(message)        

    def to_dict(self):

        return {
            "message": self.message,
            "error_code": self.error_code,
            "status_code": self.status_code,
            "context": self.context,
            "orignal_exception": str(self.original_exception) if self.original_exception else None
        }
    
    def log(self, level="error"):
        logger = get_logger("saletech.errors")

        log_func = getattr(logger, level, logger.error)

        log_func(
            "exception_raised",
            error_code=self.error_code,
            message=self.message,
            status_code=self.status_code,
            context=self.context,
            original_exception=str(self.original_exception)
            if self.original_exception else None,
        )

class SessionExpiredError(SaleTechException):
    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session {session_id} Expired",
            error_code="SESSION_EXPIRED",
            status_code=401,
            context={"session_id": session_id}
        )


class SessionNotFoundError(SaleTechException):
    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session {session_id} not found",
            error_code="SESSION_NOT_FOUND",
            status_code=404,
            context={"session_id": session_id}
        )

class ValidationError(SaleTechException):
    def __init__(self, message: str, context: Optional[dict] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            context=context
        )

class AuthorizationError(SaleTechException):
    def __init__(self, message: str = "Unauthorized", context: dict = None):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            context=context
        )

class AudioProcessingError(SaleTechException):
    def __init__(self, message: str, context: dict | None = None):
        super().__init__(
            message=message,
            error_code="AUDIO_PROCESSING_ERROR",
            status_code=500,
            context=context,
        )