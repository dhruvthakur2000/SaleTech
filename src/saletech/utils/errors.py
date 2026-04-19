from src.saletech.utils.logger import get_logger
from typing import Optional, Any

class SaleTechException(Exception):
    """Base exception for all SaleTech application errors."""

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        context: Optional[dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.context = context or {}
        self.original_exception = original_exception
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Return serializable error response."""
        return {
            "message": self.message,
            "error_code": self.error_code,
            "status_code": self.status_code,
            "context": self.context,
        }

    def log(self, level: str = "error") -> None:
        """Log the exception using structured logging."""
        logger = get_logger("saletech.errors")

        if not hasattr(logger, level):
            raise ValueError(f"Invalid log level: {level}")

        log_func = getattr(logger, level)

        log_func(
            "exception_raised",
            error_code=self.error_code,
            message=self.message,
            status_code=self.status_code,
            context=self.context,
            exc_info=self.original_exception or True,
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
    def __init__(
        self,
        message: str,
        context: dict | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            message=message,
            error_code="AUDIO_PROCESSING_ERROR",
            status_code=500,
            context=context,
            original_exception=original_exception,
        )
