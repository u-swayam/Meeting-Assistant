from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field


# ---- what the model must return (validated by us) ----
class ModelChange(BaseModel):
    original: str
    refined: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class ModelSegment(BaseModel):
    index: int
    refined_text: str
    changes: list[ModelChange] = Field(default_factory=list)


class ModelResponse(BaseModel):
    segments: list[ModelSegment]


# Gemini responseSchema (OpenAPI subset) mirroring ModelResponse.
GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "segments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "index": {"type": "INTEGER"},
                    "refined_text": {"type": "STRING"},
                    "changes": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "original": {"type": "STRING"},
                                "refined": {"type": "STRING"},
                                "reason": {"type": "STRING"},
                                "confidence": {"type": "NUMBER"},
                            },
                            "required": ["original", "refined", "reason", "confidence"],
                        },
                    },
                },
                "required": ["index", "refined_text", "changes"],
            },
        }
    },
    "required": ["segments"],
}


# ---- final audited output ----
class Change(BaseModel):
    original: str
    refined: str
    reason: str
    confidence: float


class RefinedSegment(BaseModel):
    start: float
    end: float
    speaker: str
    original_text: str
    refined_text: str
    changed: bool
    changes: list[Change] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)  # original turn metadata, passed through


class RefinementResult(BaseModel):
    model_config = ConfigDict(frozen=False)
    stage: str = "llm1_transcript_refinement"
    provider: str
    model: str
    source_file: str
    total_segments: int
    changed_segments: int
    total_changes: int
    rejected_changes: int = 0
    speakers: list[str] = Field(default_factory=list)
    duration: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    confidence_note: str = (
        "Change confidence is a conservative indication of how strongly context "
        "supports the correction; it is not a mathematical probability. The change "
        "log documents the model's interventions, not that each one is correct."
    )
    segments: list[RefinedSegment]


def derive_meta(segments, speakers=None, duration=None) -> tuple[list[str], float]:
    """Aggregate metadata from the ACTUAL segments (never from the LLM): speakers = unique non-empty speaker labels in
    order of first appearance, duration = max segment end. Non-empty/positive values supplied by the caller win."""
    segs = [s if isinstance(s, dict) else s.model_dump() for s in segments]
    spk = [str(x) for x in (speakers or []) if x]
    if not spk:
        spk = list(dict.fromkeys(str(s["speaker"]) for s in segs if s.get("speaker")))
    dur = float(duration or 0.0)
    if dur <= 0:
        dur = max([float(s.get("end") or 0.0) for s in segs] or [0.0])
    return spk, dur
