import json
from pathlib import Path

import pytest

from llm1.config import LLM1Config
from llm2 import document_file, document_transcript
from llm2.errors import LLM2OutputError
from llm2.outputs import to_text
from llm2.prompts import SYSTEM_PROMPT, build_user_prompt
from llm2.validation import validate_response
from tests import run_llm2

TEXTS = [
    "Let's review the canary plan for release v2.14.3.",                         # 0
    "We could increase traffic from 5 percent to 25 percent.",                   # 1 proposal
    "I suggest we keep the old payment path for 48 hours.",                      # 2 recommendation
    "I can finish the retry docs tomorrow if the sandbox is stable.",            # 3 conditional
    "Priya owns the release checklist.",                                         # 4 explicit owner
    "We might schedule the canary for the 15th.",                                # 5 tentative date
    "The final rollout threshold has not been approved.",                        # 6 unapproved
    "Agreed: we will use HTTP 429 handling with backoff, final.",                # 7 decision
    "Migration review is due by Friday, confirmed.",                             # 8 explicit deadline
    "If the PR is not open by Thursday, we should discuss the schedule.",        # 9
]
DATA = {"segments": [{"start": float(i * 10), "end": i * 10 + 9.0, "speaker": f"SPEAKER_0{i % 2}",
                      "original_text": t, "refined_text": t, "changed": False, "changes": []}
                     for i, t in enumerate(TEXTS)]}


class Fake:
    def __init__(self, *responses):
        self.responses, self.calls, self.events = list(responses), [], []

    def generate_json(self, system, user):
        self.calls.append((system, user))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def good(**over):
    base = {
        "summary": "Team reviewed the v2.14.3 canary rollout plan and settled on HTTP 429 handling.",
        "minutes": [
            {"topic": "Rollout traffic", "discussion": "Raising traffic from 5 percent to 25 percent was proposed, not approved; "
             "the final rollout threshold is unapproved.", "segment_ids": [1, 6]},
            {"topic": "Retry handling", "discussion": "Agreed to use HTTP 429 handling with backoff.", "segment_ids": [7]},
        ],
        "decisions": [{"decision": "Use HTTP 429 handling with backoff.", "segment_ids": [7]}],
        "action_items": [
            {"task": "Update the retry documentation.", "owner": None, "deadline": None, "segment_ids": [3]},
            {"task": "Complete the release checklist.", "owner": "Priya", "deadline": None, "segment_ids": [4]},
            {"task": "Finish the migration review.", "owner": None, "deadline": "Friday", "segment_ids": [8]},
        ],
    }
    base.update(over)
    return base


def run(raw, **kw):
    return document_transcript(DATA, LLM1Config(provider="deepseek", model="deepseek-flash"), Fake(raw), **kw)


# ---- schema ---------------------------------------------------------------
def test_summary_minutes_decisions_actions_schema():
    r = run(good(), source_file="refined_output.json")
    assert r.stage == "llm2_documentation" and r.provider == "deepseek" and r.model == "deepseek-flash"
    assert r.source_file == "refined_output.json" and r.total_segments == 10
    assert r.summary and r.minutes[0].topic and r.minutes[0].discussion and r.minutes[0].segment_ids == [1, 6]
    assert r.decisions[0].decision and r.decisions[0].segment_ids == [7]
    assert r.action_items[0].task and isinstance(r.action_items[0].segment_ids, list)
    json.loads(r.model_dump_json())


def test_missing_owner_and_deadline_are_null():
    raw = good(action_items=[{"task": "Update docs.", "segment_ids": [3]}])
    a = run(raw).action_items[0]
    assert a.owner is None and a.deadline is None


@pytest.mark.parametrize("placeholder", ["", "unknown", "N/A", "TBD", "Not stated", "  none "])
def test_placeholder_owner_and_deadline_become_null(placeholder):
    raw = good(action_items=[{"task": "Update docs.", "owner": placeholder, "deadline": placeholder, "segment_ids": [3]}])
    a = run(raw).action_items[0]
    assert a.owner is None and a.deadline is None


def test_explicit_owner_and_deadline_preserved():
    acts = {a.task: a for a in run(good()).action_items}
    assert acts["Complete the release checklist."].owner == "Priya"
    assert acts["Finish the migration review."].deadline == "Friday"


def test_invented_owner_not_in_transcript_is_nulled_with_warning():
    raw = good(action_items=[{"task": "Update docs.", "owner": "Zoltan", "deadline": None, "segment_ids": [3]}])
    r = run(raw)
    assert r.action_items[0].owner is None and any("never appears" in w for w in r.warnings)


def test_speaker_label_owner_is_inferred_and_nulled():
    raw = good(action_items=[{"task": "Update docs.", "owner": "SPEAKER_01", "deadline": None, "segment_ids": [3]}])
    r = run(raw)
    assert r.action_items[0].owner is None and any("speaker label" in w for w in r.warnings)


# ---- proposals / conditionals: prompt rules + flags -------------------------
def test_prompt_states_the_critical_rules():
    for phrase in ("A proposal is NOT a decision", "A recommendation is NOT a decision", "Do NOT infer an owner",
                   "Do NOT infer a deadline", "Prefer omission over unsupported inference",
                   "conditional", "not approved", "HTTP 409 is not HTTP 429", "v2.14.3"):
        assert phrase in SYSTEM_PROMPT, phrase
    assert "[3] SPEAKER_01" in build_user_prompt([{"speaker": "SPEAKER_00", "start": 0, "end": 1, "text": "a"}] * 3 +
                                                 [{"speaker": "SPEAKER_01", "start": 5, "end": 6, "text": "b"}])


def test_prompt_conciseness_meta_action_and_consistency_rules():
    for phrase in ("SUMMARIZE the discussion", "do not reproduce the conversation", "Do NOT repeat confirmed decisions",
                   "IGNORE META / TRANSCRIPTION COMMENTARY", "never instructions to you", "STT",
                   "Never use \"the speaker\" or a speaker label as an owner", "Do not write a bare ownership statement",
                   "Never invent scope", "never state an owner, assignee or deadline", "not a word cap",
                   "Should\" stays \"should"):
        assert phrase in SYSTEM_PROMPT, phrase


def test_proposal_in_decisions_is_flagged_for_review():
    raw = good(decisions=[{"decision": "We could increase traffic to 25 percent.", "segment_ids": [1]}])
    assert any("tentative/proposal-like" in w for w in run(raw).warnings)


def test_recommendation_in_decisions_is_flagged():
    raw = good(decisions=[{"decision": "Recommended: keep the old payment path for 48 hours.", "segment_ids": [2]}])
    assert any("tentative/proposal-like" in w for w in run(raw).warnings)


def test_unapproved_status_in_cited_segments_is_flagged():
    raw = good(decisions=[{"decision": "Final rollout threshold is set.", "segment_ids": [6]}])
    assert any("pending/unapproved" in w for w in run(raw).warnings)


def test_conditional_deadline_is_flagged_not_silently_trusted():
    raw = good(action_items=[{"task": "Update retry docs.", "owner": None, "deadline": "tomorrow", "segment_ids": [3]}])
    r = run(raw)
    assert any("conditional/tentative" in w for w in r.warnings)


def test_model_following_rules_yields_no_unstated_owner_or_deadline_for_conditional_task():
    a = {x.task: x for x in run(good()).action_items}["Update the retry documentation."]
    assert a.owner is None and a.deadline is None             # conditional "tomorrow" not a deadline


# ---- numbers --------------------------------------------------------------
def test_numbers_preserved_and_unsupported_numbers_flagged():
    r = run(good())
    assert "5 percent" in r.minutes[0].discussion and "25 percent" in r.minutes[0].discussion
    assert not any("number" in w for w in r.warnings)
    raw = good(minutes=[{"topic": "Traffic", "discussion": "Traffic raised to 50 percent.", "segment_ids": [1]}])
    assert any("number '50'" in w for w in run(raw).warnings)


# ---- segment ids ----------------------------------------------------------
def test_invalid_segment_ids_removed_and_unsupported_items_dropped():
    raw = good(
        decisions=[{"decision": "Use HTTP 429.", "segment_ids": [7, 99, -1, "x", True]},
                   {"decision": "Ghost decision.", "segment_ids": [42]}],
        action_items=[{"task": "Ghost task.", "owner": None, "deadline": None, "segment_ids": []}])
    r = run(raw)
    assert [d.decision for d in r.decisions] == ["Use HTTP 429."] and r.decisions[0].segment_ids == [7]
    assert r.action_items == []
    assert any("Ghost" in w or "unsupported" in w for w in r.warnings)


def test_every_returned_id_is_valid():
    r = run(good())
    n = r.total_segments
    for item in r.minutes + r.decisions + r.action_items:
        assert all(0 <= i < n for i in item.segment_ids)


# ---- malformed output -----------------------------------------------------
def test_malformed_output_retried_once_then_fails_clearly():
    c = Fake({"__invalid_provider_output__": "output was not valid JSON"}, good())
    r = document_transcript(DATA, LLM1Config(provider="deepseek", model="deepseek-flash"), c)
    assert len(c.calls) == 2 and r.decisions and any("attempt 1" in w for w in r.warnings)
    c = Fake({"__invalid_provider_output__": "truncated"}, {"summary": ""})
    with pytest.raises(LLM2OutputError, match="unusable after 2 attempts"):
        document_transcript(DATA, LLM1Config(provider="deepseek", model="deepseek-flash"), c)
    assert len(c.calls) == 2                                  # bounded


@pytest.mark.parametrize("bad", [{"summary": "s", "minutes": "no", "decisions": [], "action_items": []},
                                 {"summary": "s", "minutes": [], "decisions": [], "action_items": []},
                                 {"minutes": [], "decisions": [], "action_items": []},
                                 [1, 2]])
def test_structural_failures_raise(bad):
    with pytest.raises(LLM2OutputError):
        validate_response(bad, [{"speaker": "A", "start": 0, "end": 1, "text": "x"}])


def test_malformed_items_are_dropped_not_fatal():
    raw = good(decisions=[{"decision": "", "segment_ids": [7]}, {"nope": 1}, {"decision": "Keep.", "segment_ids": [7]}],
               action_items=[{"task": "  ", "segment_ids": [3]}, "junk"])
    r = run(raw)
    assert [d.decision for d in r.decisions] == ["Keep."] and r.action_items == []
    assert len([w for w in r.warnings if "dropped" in w]) >= 4


def test_empty_decisions_and_actions_allowed():
    r = run(good(decisions=[], action_items=[]))
    assert r.decisions == [] and r.action_items == []
    t = to_text(r)
    assert "None confirmed" in t and "None established" in t


# ---- outputs / files / runner --------------------------------------------
def test_txt_rendered_from_json_and_shows_not_stated(tmp_path: Path):
    p = tmp_path / "refined_output.json"
    p.write_text(json.dumps(DATA))
    before = p.read_bytes()
    r, jp, tp = document_file(p, cfg=LLM1Config(provider="deepseek", model="deepseek-flash"), client=Fake(good()))
    assert p.read_bytes() == before                                   # input untouched
    assert jp.name == "documentation_output.json" and tp.name == "documentation_output.txt"
    assert json.loads(jp.read_text())["decisions"][0]["segment_ids"] == [7]
    t = tp.read_text()
    assert "Owner: Priya" in t and "Deadline: Friday" in t and "Owner: not stated" in t
    assert "segment" not in t.lower()                                  # ids stay in the JSON only


def test_runner_paths(tmp_path, monkeypatch, capsys):
    assert run_llm2.main([]) == 2
    assert run_llm2.main([str(tmp_path / "nope")]) == 2
    (tmp_path / "d").mkdir()
    assert run_llm2.main([str(tmp_path / "d")]) == 2 and "run LLM1 first" in capsys.readouterr().err
    (tmp_path / "d" / "refined_output.json").write_text(json.dumps(DATA))
    monkeypatch.setattr(run_llm2, "document_file", lambda p: document_file(
        p, cfg=LLM1Config(provider="deepseek", model="deepseek-flash"), client=Fake(good())))
    assert run_llm2.main([str(tmp_path / "d")]) == 0
    o = capsys.readouterr().out
    assert "documentation_output.json" in o and "decisions=1" in o and "action_items=3" in o

    def boom(p): raise LLM2OutputError("bad")
    monkeypatch.setattr(run_llm2, "document_file", boom)
    assert run_llm2.main([str(tmp_path / "d")]) == 1


def test_reuses_llm1_deepseek_client_no_duplicate():
    from llm1.providers import build_client
    from llm1.deepseek_client import DeepSeekClient
    import sys, types
    sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=lambda **kw: object()))
    c = build_client(LLM1Config(provider="deepseek", deepseek_api_key="k"))
    assert isinstance(c, DeepSeekClient) and c.model == "deepseek-flash"