from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class Word(BaseModel):
    model_config = ConfigDict(frozen=True)
    start: float
    end: float
    text: str
    confidence: Optional[float] = None


class STTSegment(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: int
    start: float
    end: float
    text: str
    confidence: Optional[float] = None
    avg_logprob: Optional[float] = None        # Whisper-style providers only
    no_speech_prob: Optional[float] = None     # Whisper-style providers only
    compression_ratio: Optional[float] = None  # Whisper-style providers only
    chunk_index: int = 0


class STTResult(BaseModel):
    """Raw STT output. Frozen: downstream stages must not modify it.
    raw_response holds the provider's untouched JSON."""
    model_config = ConfigDict(frozen=True)
    text: str
    language: Optional[str] = None
    duration: float = 0.0
    segments: tuple[STTSegment, ...] = ()
    words: tuple[Word, ...] = ()
    provider: str = "unknown"
    model: str = "unknown"
    audio_path: Optional[str] = None
    warnings: tuple[str, ...] = ()
    raw_response: list[dict[str, Any]] = Field(default_factory=list, repr=False)
