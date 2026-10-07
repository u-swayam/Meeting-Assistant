from __future__ import annotations
from pydantic import BaseModel, ConfigDict


class DiarSegment(BaseModel):
    model_config = ConfigDict(frozen=True)
    start: float
    end: float
    speaker: str


class DiarizationResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    segments: tuple[DiarSegment, ...] = ()            # may overlap
    exclusive_segments: tuple[DiarSegment, ...] = ()  # one speaker at a time (if available)
    speakers: tuple[str, ...] = ()
    model: str = ""
    device: str = ""
    audio_path: str | None = None
    warnings: tuple[str, ...] = ()
