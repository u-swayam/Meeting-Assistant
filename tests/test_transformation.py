import hashlib
import json
from pathlib import Path

import pytest

from llm1.config import LLM1Config
from llm2 import document_file, document_transcript
from llm2.guidance import GUIDANCE_ADDENDUM, cross_check, render_guidance
from transformation import transform_file, transform_transcript
from transformation.errors import TransformationError
from transformation.validation import validate_response
from tests import run_transformation

CFG = LLM1Config(provider="deepseek", model="deepseek-flash")
TEXTS = [
    "We could move the ledger to PostgreSQL.",                       # 0 proposal
    "Sounds good to me.",                                            # 1 acceptance
    "Okay, let's use PostgreSQL. That is the decision.",             # 2 confirmation
    "We could use Redis for sessions.",                              # 3 proposal
    "Maybe. Let's discuss it next week.",                            # 4 deferral
    "Priya owns the release checklist.",                             # 5 explicit owner
    "If the tests pass, we'll deploy on Friday.",                    # 6 conditional
    "Someone should update the retry docs.",                         # 7 no owner
]
DATA = {"segments": [{"start": float(i * 5), "end": i * 5 + 4.0, "speaker": f"SPEAKER_0{i % 3}",
                      "original_text": t, "refined_text": t, "changed": False, "changes": []}
                     for i, t in enumerate(TEXTS)]}


def good():
    return {
        "topics": [{"topic_id": "t1", "title": "Ledger database", "segment_ids": [0, 1, 2]},
                   {"topic_id": "t2", "title": "Sessions cache", "segment_ids": [3, 4]},
                   {"topic_id": "t3", "title": "Release", "segment_ids": [5, 6, 7]}],
        "discussion_units": [{"discussion_id": "d1", "topic_id": "t1", "segment_ids": [0, 1, 2], "type": "decision"}],
        "decision_candidates": [
            {"decision_id": "x", "topic_id": "t1", "segment_ids": [0, 1, 2], "status": "confirmed", "text": "Use PostgreSQL."},
            {"decision_id": "y", "topic_id": "t2", "segment_ids": [3, 4], "status": "unresolved", "text": "Use Redis for sessions."}],
        "action_candidates": [
            {"action_id": "a", "topic_id": "t3", "segment_ids": [5], "status": "confirmed",
             "description": "Complete the release checklist.", "owner": "Priya", "deadline": None},
            {"action_id": "b", "topic_id": "t3", "segment_ids": [6], "status": "conditional",
             "description": "Deploy.", "owner": None, "deadline": None, "condition": "if the tests pass"},
            {"action_id": "c", "topic_id": "t3", "segment_ids": [7], "status": "proposed",
             "description": "Update the retry docs.", "owner": None, "deadline": None}],
        "relations": [{"type": "accepts", "from_segment": 1, "to_segment": 0}],
    }


class Fake:
    def __init__(self, *resp):
        self.resp, self.calls, self.events = list(resp), [], []

    def generate_json(self, system, user):
        self.calls.append((system, user))
        return self.resp.pop(0)


def T(raw=None):
    return transform_transcript(DATA, CFG, Fake(raw or good()), source_file="refined_output.json")


# A / I / J: schema, evidence ids, topics
def test_valid_schema_ids_and_topics():
    r = T()
    assert r.stage == "transformation" and r.source_file == "refined_output.json" and r.total_segments == 8
    assert [t.topic_id for t in r.topics] == ["topic_1", "topic_2", "topic_3"]
    assert r.topics[0].segment_ids == [0, 1, 2] and r.topics[0].title == "Ledger database"
    assert r.decision_candidates[0].segment_ids == [0, 1, 2] and r.decision_candidates[0].topic_id == "topic_1"
    assert r.decision_candidates[0].decision_id == "decision_1" and r.action_candidates[1].action_id == "action_2"
    assert r.discussion_units[0].topic_id == "topic_1" and r.relations[0].type == "accepts"
    json.loads(r.model_dump_json())
    assert not r.warnings


def test_noncontiguous_topic_allowed_and_ordered():
    raw = good()
    raw["topics"] = [{"topic_id": "b", "title": "Later", "segment_ids": [5, 6, 7]},
                     {"topic_id": "a", "title": "Ledger", "segment_ids": [0, 1, 2, 6]}]
    r = T(raw)
    assert [t.title for t in r.topics] == ["Ledger", "Later"] and r.topics[0].segment_ids == [0, 1, 2, 6]


# B: malformed output
@pytest.mark.parametrize("bad", [{"__invalid_provider_output__": "truncated"}, [1], {"topics": []},
                                 {"topics": "x", "decision_candidates": [], "action_candidates": []},
                                 {"topics": [{"title": "", "segment_ids": [0]}], "decision_candidates": [], "action_candidates": []}])
def test_malformed_output_rejected(bad):
    with pytest.raises(TransformationError):
        validate_response(bad, [{"speaker": "A", "start": 0, "end": 1, "text": "x"}])


def test_retry_once_then_fail_clearly():
    c = Fake({"__invalid_provider_output__": "x"}, good())
    assert transform_transcript(DATA, CFG, c).topics and len(c.calls) == 2
    c = Fake({"__invalid_provider_output__": "x"}, {"topics": []})
    with pytest.raises(TransformationError, match="after 2 attempts"):
        transform_transcript(DATA, CFG, c)
    assert len(c.calls) == 2


# C: invalid ids
def test_invalid_segment_ids_removed_and_unsupported_items_dropped():
    raw = good()
    raw["decision_candidates"] = [{"decision_id": "x", "segment_ids": [2, 99, -1, "z", True], "status": "confirmed", "text": "Use PostgreSQL."},
                                  {"decision_id": "y", "segment_ids": [42], "status": "confirmed", "text": "Ghost."}]
    raw["action_candidates"] = [{"action_id": "a", "segment_ids": [], "status": "confirmed", "description": "Ghost task."}]
    raw["relations"] = [{"type": "accepts", "from_segment": 1, "to_segment": 99}]
    r = T(raw)
    assert [d.text for d in r.decision_candidates] == ["Use PostgreSQL."] and r.decision_candidates[0].segment_ids == [2]
    assert r.action_candidates == [] and r.relations == [] and any("dropped" in w for w in r.warnings)


# D / E: proposal vs confirmed (status is kept as the model gave it; invalid statuses dropped)
def test_proposal_stays_proposal_and_confirmation_is_confirmed():
    r = T()
    st = {d.text: d.status for d in r.decision_candidates}
    assert st["Use PostgreSQL."] == "confirmed" and st["Use Redis for sessions."] == "unresolved"
    raw = good(); raw["decision_candidates"][0]["status"] = "agreed!"
    assert not any(d.text == "Use PostgreSQL." for d in T(raw).decision_candidates)


def test_confirmed_with_unapproved_wording_is_flagged():
    raw = good()
    raw["decision_candidates"] = [{"segment_ids": [4], "status": "confirmed", "text": "Use Redis.", "decision_id": "z"}]
    data = {"segments": [dict(s) for s in DATA["segments"]]}
    data["segments"][4]["refined_text"] = "The final choice has not been approved yet."
    r = transform_transcript(data, CFG, Fake(raw))
    assert any("pending/unapproved" in w for w in r.warnings)


# F / G / H: owner, deadline, condition
def test_owner_deadline_null_and_condition_preserved():
    a = {x.description: x for x in T().action_candidates}
    assert a["Update the retry docs."].owner is None and a["Update the retry docs."].deadline is None
    assert a["Complete the release checklist."].owner == "Priya" and a["Complete the release checklist."].deadline is None
    assert a["Deploy."].status == "conditional" and a["Deploy."].condition == "if the tests pass"


def test_invented_owner_and_placeholders_nulled():
    raw = good()
    raw["action_candidates"] = [
        {"segment_ids": [7], "status": "proposed", "description": "Update docs.", "owner": "Zoltan", "deadline": "unknown"},
        {"segment_ids": [7], "status": "proposed", "description": "Update docs 2.", "owner": "SPEAKER_01", "deadline": ""},
        {"segment_ids": [6], "status": "conditional", "description": "Deploy."}]
    r = T(raw)
    assert all(a.owner is None and a.deadline is None for a in r.action_candidates)
    assert any("never appears" in w for w in r.warnings) and any("speaker label" in w for w in r.warnings)
    assert any("without a stated condition" in w for w in r.warnings)


# K / L: LLM2 gets BOTH, transcript stays authoritative
LLM2_OUT = {"summary": "s", "minutes": [{"topic": "Ledger", "discussion": "PostgreSQL chosen.", "segment_ids": [0, 1, 2]}],
            "decisions": [{"decision": "Use PostgreSQL.", "segment_ids": [0, 1, 2]}],
            "action_items": [{"task": "Complete the release checklist.", "owner": "Priya", "deadline": None, "segment_ids": [5]}]}


def test_llm2_receives_transcript_and_transformation_and_authority_rules():
    trans = T().model_dump(mode="json")
    c = Fake(LLM2_OUT)
    r = document_transcript(DATA, CFG, c, source_file="refined_output.json", transformation=trans,
                            transformation_source="transformation_output.json")
    system, user = c.calls[0]
    assert "[2] SPEAKER_02" in user and "Okay, let's use PostgreSQL" in user          # refined transcript present
    assert "decision_1 [confirmed]" in user and "action_2 [conditional]" in user and "(segments [0, 1, 2])" in user
    assert GUIDANCE_ADDENDUM in system and "The transcript is the only source of truth" in system
    assert "Verify EVERY candidate against the transcript" in system and "Never copy an owner or deadline" in system
    assert r.transformation_source == "transformation_output.json"


def test_llm2_unguided_is_unchanged_and_backward_compatible():
    from llm2.prompts import SYSTEM_PROMPT
    c = Fake(LLM2_OUT)
    r = document_transcript(DATA, CFG, c)
    assert c.calls[0][0] == SYSTEM_PROMPT and "STRUCTURED GUIDANCE" not in c.calls[0][1] and r.transformation_source is None


def test_cross_check_flags_disagreements_without_editing():
    trans = T().model_dump(mode="json")
    out = dict(LLM2_OUT)
    out["decisions"] = LLM2_OUT["decisions"] + [{"decision": "Use Redis.", "segment_ids": [3, 4]}]
    out["action_items"] = [{"task": "Deploy on Friday.", "owner": "Priya", "deadline": "Friday", "segment_ids": [6]}]
    r = document_transcript(DATA, CFG, Fake(out), transformation=trans)
    w = " | ".join(r.warnings)
    assert "unresolved, not confirmed" in w and "owner 'Priya' not present in guidance" in w
    assert "deadline 'Friday' not present in guidance" in w and "marked this action conditional" in w
    assert len(r.decisions) == 2                       # flagged, not silently removed or trusted


def test_existing_llm2_validation_still_applies_with_guidance():
    trans = T().model_dump(mode="json")
    out = dict(LLM2_OUT)
    out["action_items"] = [{"task": "x", "owner": "Zoltan", "deadline": None, "segment_ids": [99]}]
    r = document_transcript(DATA, CFG, Fake(out), transformation=trans)
    assert r.action_items == []                        # invalid segment id -> unsupported -> dropped, as before


def test_render_guidance_contains_ids_status_and_nulls():
    g = render_guidance(T().model_dump(mode="json"))
    assert "topic_1: Ledger database" in g and "owner=None deadline=None" in g and "condition=if the tests pass" in g


# M / O: file-level behaviour and end to end
def test_failure_does_not_touch_refined_or_write_output(tmp_path: Path):
    p = tmp_path / "refined_output.json"; p.write_text(json.dumps(DATA))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(TransformationError):
        transform_file(p, cfg=CFG, client=Fake({"__invalid_provider_output__": "x"}, {"__invalid_provider_output__": "x"}))
    assert hashlib.sha256(p.read_bytes()).hexdigest() == h and not (tmp_path / "transformation_output.json").exists()
    assert not (tmp_path / "documentation_output.json").exists()


def test_end_to_end_transform_then_document_autodetects(tmp_path: Path):
    p = tmp_path / "refined_output.json"; p.write_text(json.dumps(DATA))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    res, tp = transform_file(p, cfg=CFG, client=Fake(good()))
    assert tp.name == "transformation_output.json" and json.loads(tp.read_text())["decision_candidates"][0]["segment_ids"] == [0, 1, 2]
    c = Fake(LLM2_OUT)
    doc, jp, txt = document_file(p, cfg=CFG, client=c)
    assert doc.transformation_source == "transformation_output.json" and "decision_1" in c.calls[0][1]
    d = json.loads(jp.read_text())
    assert d["decisions"][0]["segment_ids"] == [0, 1, 2] and d["action_items"][0]["owner"] == "Priya"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == h and txt.is_file()
    for k in ("summary", "minutes", "decisions", "action_items", "warnings"):
        assert k in d


def test_runner_and_pipeline_and_server_integration(tmp_path, monkeypatch, capsys):
    assert run_transformation.main([]) == 2 and run_transformation.main([str(tmp_path)]) == 2
    from app import pipeline_runner, server
    assert pipeline_runner.STAGES[3:] == ["LLM1 Refinement", "Transformation", "LLM2 Documentation"]
    m = tmp_path / "m1"; m.mkdir()
    (m / "final_output.json").write_text(json.dumps({"turns": []}))
    (m / "refined_output.json").write_text(json.dumps(DATA))
    transform_file(m / "refined_output.json", cfg=CFG, client=Fake(good()))
    monkeypatch.setattr(server, "OUTPUTS", tmp_path)
    assert server.list_meetings()[0]["has_transformation"] is True
    assert server.bundle("m1")["transformation"]["stage"] == "transformation"
