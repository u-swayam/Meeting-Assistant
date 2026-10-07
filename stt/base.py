from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from .schemas import STTResult


class STTProvider(ABC):
    """Any STT backend implements this; nothing downstream depends on the provider."""
    name: str = "base"

    @abstractmethod
    def transcribe(self, audio_path: str | Path) -> STTResult:
        """Raise AudioError subclasses for bad audio, STTError subclasses for provider failures."""
