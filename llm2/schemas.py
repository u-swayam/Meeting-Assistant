from __future__ import annotations
from pydantic import BaseModel, Field


class MinuteItem(BaseModel):
    topic: str
    discussion: str
    segment_ids: list[int] = Field(default_factory=list)


class DecisionItem(BaseModel):
    decision: str
    segment_ids: list[int] = Field(default_factory=list)


class ActionItem(BaseModel):
    task: str
    owner: str | None = None
    deadline: str | None = None
    segment_ids: list[int] = Field(default_factory=list)


class DocumentationResult(BaseModel):
    stage: str = "llm2_documentation"
    provider: str
    model: str
    source_file: str
    total_segments: int
    summary: str
    minutes: list[MinuteItem]
    decisions: list[DecisionItem]
    action_items: list[ActionItem]
    warnings: list[str] = Field(default_factory=list)
    transformation_source: str | None = None     # set only when guided by transformation_output.json
