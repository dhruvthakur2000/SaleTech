"""
SaleTech Structured Logging
Production-grade logging with context
"""
import structlog
import logging
import sys
from datetime import datetime


def setup_logging(log_level: str = "INFO"):
    """Configure structured logging"""
    
    # Log file path with date
    from logging.handlers import RotatingFileHandler
    import os
    # Force logs to root logs directory
    from datetime import datetime
    log_dir = r"c:\\SaleTech\\logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y-%m-%d___%H-%M-%S')}.log")
    print(f"[Logger Debug] Log file path: {log_file}")
    print(f"[Logger Debug] Log file path: {log_file}")

    # File handler (rotates at 10MB, keeps 5 backups)
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(message)s'))

    # Stream handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(message)s'))

    # Root logger setup: clear all handlers, then add ours
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    # Ensure all loggers inherit handlers from root and propagate
    logging.lastResort = None  # Disable fallback handler
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level.upper()))
        logger.propagate = True
    
    # Configure structlog to use stdlib logger so logs go through standard logging handlers
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance"""
    return structlog.get_logger(name)


class SessionLogger:
    """Session-aware logger that includes session context"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.logger = get_logger("saletech.session")
        self.logger = self.logger.bind(session_id=session_id)
    
    def info(self, event: str, **kwargs):
        """Log info message"""
        self.logger.info(event, **kwargs)
    
    def error(self, event: str, **kwargs):
        """Log error message"""
        self.logger.error(event, **kwargs)
    
    def warning(self, event: str, **kwargs):
        """Log warning message"""
        self.logger.warning(event, **kwargs)
    
    def debug(self, event: str, **kwargs):
        """Log debug message"""
        self.logger.debug(event, **kwargs)
    
    def latency(self, component: str, latency_ms: float, **kwargs):
        """Log latency metric"""
        self.logger.info(
            "latency_metric",
            component=component,
            latency_ms=latency_ms,
            **kwargs
        )
    
    def state_transition(self, from_state: str, to_state: str, **kwargs):
        """Log state transition"""
        self.logger.info(
            "state_transition",
            from_state=from_state,
            to_state=to_state,
            **kwargs
        )
    
    def error_with_context(self, event: str, error: Exception, **kwargs):
        """Log error with exception context"""
        self.logger.error(
            event,
            error_type=type(error).__name__,
            error_message=str(error),
            **kwargs,
            exc_info=True
        )


# Application-wide loggers
app_logger = get_logger("saletech.app")
api_logger = get_logger("saletech.api")
audio_logger = get_logger("saletech.audio")
model_logger = get_logger("saletech.models")