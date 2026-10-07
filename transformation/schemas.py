from __future__ import annotations
from typing import Literal

from pydantic import BaseModel, Field

DecisionStatus = Literal["confirmed", "tentative", "proposed", "unresolved"]
ActionStatus = Literal["confirmed", "proposed", "conditional", "tentative"]
UnitType = Literal["discussion", "proposal", "decision", "action", "question", "clarification", "status_update"]
RelationType = Literal["supports", "accepts", "rejects", "defers", "elaborates", "responds_to"]


class Topic(BaseModel):
    topic_id: str
    title: str
    segment_ids: list[int]


class DiscussionUnit(BaseModel):
    discussion_id: str
    topic_id: str | None = None
    segment_ids: list[int]
    type: UnitType = "discussion"


ConsensusClass = Literal["confirmed", "likely", "proposed", "unresolved", "contested"]
ReconstructionSource = Literal["model", "candidate_text", "original_text", "none"]


class ConsensusEvidence(BaseModel):
    """Segment ids ONLY. An empty list means no such evidence exists (nothing is manufactured)."""
    decision_statement: list[int] = Field(default_factory=list)
    supporting_statements: list[int] = Field(default_factory=list)
    explicit_agreement: list[int] = Field(default_factory=list)
    explicit_finalization: list[int] = Field(default_factory=list)
    opposition: list[int] = Field(default_factory=list)
    unresolved: list[int] = Field(default_factory=list)


class Consensus(BaseModel):
    """Additional evidence layer. Does NOT replace DecisionCandidate.status. `score` is a heuristic internal
    confidence derived from the evidence segments, not a calibrated probability."""
    classification: ConsensusClass
    level: Literal["high", "medium", "low"]
    score: float
    score_kind: str = "heuristic_internal_confidence_not_calibrated"
    evidence: ConsensusEvidence
    notes: list[str] = Field(default_factory=list)


class DecisionCandidate(BaseModel):
    decision_id: str
    topic_id: str | None = None
    segment_ids: list[int]
    status: DecisionStatus
    text: str
    # --- additive, optional (absent in older transformation_output.json files) ---
    consensus: Consensus | None = None
    trigger_segment: int | None = None
    context_segments: list[int] = Field(default_factory=list)
    original_text: str | None = None            # refined text of trigger_segment, verbatim
    reconstructed_text: str | None = None       # self-contained, grounded in context_segments only
    reconstruction_source: ReconstructionSource | None = None


class ActionCandidate(BaseModel):
    action_id: str
    topic_id: str | None = None
    segment_ids: list[int]
    status: ActionStatus
    description: str
    owner: str | None = None
    deadline: str | None = None
    condition: str | None = None          # kept when the action is conditional ("if the tests pass")


class Relation(BaseModel):
    type: RelationType
    from_segment: int
    to_segment: int


class TransformationResult(BaseModel):
    stage: str = "transformation"
    provider: str
    model: str
    source_file: str
    total_segments: int
    topics: list[Topic]
    discussion_units: list[DiscussionUnit] = Field(default_factory=list)
    decision_candidates: list[DecisionCandidate] = Field(default_factory=list)
    action_candidates: list[ActionCandidate] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
