from abc import ABC, abstractmethod
from typing import Any

class STTModel(ABC):
    """Abstract base class for all STT models."""

    @abstractmethod
    def transcribe(self, audio_path: str, **kwargs) -> str:
        """Transcribe the given audio file and return the predicted text."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the model."""
        pass

    @property
    def version(self) -> str:
        """Return the version of the model (optional)."""
        return "unknown"
