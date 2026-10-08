"""Deterministic, fixture-based tests for consensus analysis + decision context reconstruction (no model calls)."""
import pytest

from app.evidence import build_evidence
from llm1.config import LLM1Config
from llm2 import document_transcript
from llm2.guidance import GUIDANCE_ADDENDUM, cross_check, render_guidance
from transformation import transform_transcript
from transformation.consensus import analyze_consensus, cues, select_context
from transformation.reconstruction import reconstruct
from transformation.validation import validate_response

CFG = LLM1Config(provider="deepseek", model="deepseek-flash")


def segs(*rows):
    return [{"speaker": s, "start": float(i * 5), "end": i * 5 + 4.0, "text": t} for i, (s, t) in enumerate(rows)]


def data(S):
    return {"segments": [{"start": s["start"], "end": s["end"], "speaker": s["speaker"], "original_text": s["text"],
                          "refined_text": s["text"], "changed": False, "changes": []} for s in S]}


def run(S, ids, status="confirmed", text="Use PostgreSQL as the ledger.", **extra):
    raw = {"topics": [{"topic_id": "t", "title": "T", "segment_ids": list(range(len(S)))}], "decision_candidates":
           [dict({"segment_ids": ids, "status": status, "text": text}, **extra)], "action_candidates": []}
    out, warns = validate_response(raw, S)
    return (out["decision_candidates"][0] if out["decision_candidates"] else None), warns


class Fake:
    def __init__(self, *r):
        self.r, self.calls, self.events = list(r), [], []

    def generate_json(self, s, u):
        self.calls.append((s, u)); return self.r.pop(0)


CONFIRMED = segs(("A", "Should we use PostgreSQL for the ledger?"), ("B", "It gives us better consistency than the document store."),
                 ("C", "Agreed."), ("A", "Okay, let's go with that."), ("B", "Next, the release schedule."))


# 1 confirmed with explicit agreement
def test_confirmed_with_explicit_agreement():
    d, w = run(CONFIRMED, [0, 1, 2, 3])
    c = d.consensus
    assert c.classification == "confirmed" and c.level == "high" and c.score >= 0.75
    assert c.evidence.decision_statement == [0] and 1 in c.evidence.supporting_statements
    assert 2 in c.evidence.explicit_agreement and c.evidence.opposition == [] and not w
    assert c.score_kind == "heuristic_internal_confidence_not_calibrated"
    assert d.status == "confirmed"                       # status semantics untouched


# 2 proposal, no agreement
def test_proposal_with_no_agreement():
    S = segs(("A", "I think we should move the ledger to PostgreSQL."), ("B", "Next, the release schedule."))
    d, _ = run(S, [0], status="proposed")
    assert d.consensus.classification == "proposed" and d.consensus.level == "low"
    assert d.consensus.evidence.explicit_agreement == [] and d.status == "proposed"


# 3 explicit disagreement
def test_explicit_disagreement():
    S = segs(("A", "We should use Redis for the ledger."), ("B", "I disagree. I don't think that will work."))
    d, _ = run(S, [0, 1], status="proposed", text="Use Redis for the ledger.")
    assert d.consensus.classification == "contested" and d.consensus.evidence.opposition == [1] and d.consensus.level == "low"


# 4 context-dependent final utterance
def test_context_dependent_final_utterance():
    d, _ = run(CONFIRMED, [3], reconstructed_text="The team agreed to use PostgreSQL for the ledger.")
    assert d.trigger_segment == 3 and d.original_text == "Okay, let's go with that."
    assert d.reconstructed_text == "The team agreed to use PostgreSQL for the ledger." and d.reconstruction_source == "model"
    assert d.consensus.classification == "confirmed"
    d2, _ = run(CONFIRMED, [3])                          # no model text -> deterministic fallback, still standalone
    assert d2.reconstruction_source == "candidate_text" and "PostgreSQL" in d2.reconstructed_text
    assert d2.reconstructed_text != d2.original_text


# 5 preceding segments used, bounded
def test_context_uses_preceding_segments_and_is_bounded():
    d, _ = run(CONFIRMED, [3])
    assert d.context_segments[:4] == [0, 1, 2, 3] or {0, 1, 2, 3} <= set(d.context_segments)
    long = segs(*[("A", f"Unrelated chatter number {i}.") for i in range(20)] + [("B", "Should we use PostgreSQL?"), ("C", "Yes, that works."), ("A", "Okay, let's go with that.")])
    ctx = select_context([22], long)
    assert 20 in ctx and 22 in ctx and len(ctx) <= 8 and 0 not in ctx
    self_contained = segs(("A", "x"), ("A", "We will use PostgreSQL as the authoritative ledger for all payment records."))
    assert select_context([1], self_contained) == [0, 1]


# 6 following agreement
def test_context_uses_following_agreement_and_stops_at_unrelated():
    S = segs(("A", "I think we should use PostgreSQL."), ("B", "Agreed."), ("C", "Okay, let's go with that."), ("A", "Next, the release schedule."))
    d, _ = run(S, [0])
    assert d.context_segments == [0, 1, 2] and 3 not in d.context_segments
    assert d.consensus.classification == "confirmed" and d.trigger_segment == 2


# 7 no hallucinated consensus
def test_no_hallucinated_consensus():
    S = segs(("A", "I think we should adopt Kafka."), ("B", "Let me look at the budget."), ("C", "That has not been agreed yet."))
    d, _ = run(S, [0, 1, 2], status="proposed", text="Adopt Kafka.")
    ev = d.consensus.evidence
    assert d.consensus.classification == "unresolved" and ev.explicit_agreement == [] and ev.explicit_finalization == []
    assert cues("That has not been agreed yet.")["agree"] is False and cues("I don't agree with that.")["agree"] is False
    assert cues("Nobody objected.")["agree"] is False
    S2 = segs(("A", "We could use Kafka."), ("B", "Okay."))
    assert run(S2, [0, 1], status="proposed", text="Use Kafka.")[0].consensus.classification != "confirmed"


# 8 no hallucinated facts in reconstruction
def test_reconstruction_rejects_invented_facts():
    d, w = run(CONFIRMED, [3], reconstructed_text="The team agreed to use MongoDB for the ledger.")
    assert "MongoDB" not in d.reconstructed_text and d.reconstruction_source != "model"
    assert any("reconstructed_text rejected" in x for x in w)


# 9 negation preservation
def test_negation_preserved():
    S = segs(("A", "Should we use Redis for sessions?"), ("B", "No, we will not use Redis for sessions. Agreed?"), ("A", "Agreed."))
    bad, w = run(S, [1, 2], text="Do not use Redis for sessions.", reconstructed_text="The team agreed to use Redis for sessions.")
    assert bad.reconstruction_source != "model" and any("negation" in x for x in w)
    good, _ = run(S, [1, 2], text="Do not use Redis for sessions.", reconstructed_text="The team agreed not to use Redis for sessions.")
    assert good.reconstruction_source == "model" and "not" in good.reconstructed_text
    assert "not" in bad.reconstructed_text.lower()       # fallback keeps the negation too


# 10 number / identifier preservation
def test_numbers_and_identifiers_preserved():
    S = segs(("A", "Should the token lifetime be 15 minutes on v2.14.3?"), ("B", "Yes, that works, and return HTTP 409 on conflict."), ("A", "Okay, let's go with that."))
    ok, _ = run(S, [2], text="Token lifetime is 15 minutes on v2.14.3.", reconstructed_text="The team agreed on a token lifetime of 15 minutes on v2.14.3.")
    assert ok.reconstruction_source == "model" and "v2.14.3" in ok.reconstructed_text
    for wrong in ("The team agreed on a token lifetime of 25 minutes on v2.14.3.", "The team agreed to return HTTP 429 on conflict.",
                  "The team agreed on a 15 minute lifetime on v2.14.4."):
        d, _ = run(S, [2], text="Token lifetime is 15 minutes on v2.14.3.", reconstructed_text=wrong)
        assert d.reconstruction_source != "model"


# status / uncertainty preserved in reconstruction
def test_uncertainty_and_status_preserved_in_reconstruction():
    S = segs(("A", "We might move the ledger to PostgreSQL."), ("B", "Maybe. Let's discuss it next week."))
    d, _ = run(S, [0, 1], status="unresolved", text="Move the ledger to PostgreSQL.", reconstructed_text="The team agreed to move the ledger to PostgreSQL.")
    assert d.reconstruction_source != "model" and d.reconstructed_text.startswith("Unresolved")
    assert d.consensus.classification == "unresolved"


# 11 invalid segment ids
def test_invalid_segment_ids():
    d, w = run(CONFIRMED, [3, 99, -1])
    assert d.segment_ids == [3] and all(0 <= i < len(CONFIRMED) for i in d.context_segments)
    assert run(CONFIRMED, [99])[0] is None
    assert select_context([99, -5, True], CONFIRMED) == []


# 12 empty / missing context
def test_empty_or_missing_context():
    assert reconstruct("x", "Use X.", "confirmed", "confirmed", CONFIRMED, [], 0, 0, []) == (None, "none", None)
    d, _ = run(segs(("A", "Okay.")), [0], text="Something.")
    assert d.context_segments == [0] and d.reconstructed_text is not None


# status is never silently changed; mismatch is flagged
def test_confirmed_status_with_weak_evidence_is_flagged_not_changed():
    S = segs(("A", "I think we should adopt Kafka."), ("B", "Next, the release schedule."))
    d, w = run(S, [0], status="confirmed", text="Adopt Kafka.")
    assert d.status == "confirmed" and d.consensus.level == "low" and any("conversational evidence" in x for x in w)


def test_opposition_then_explicit_agreement_is_confirmed_with_penalty():
    S = segs(("A", "We should use Redis."), ("B", "I disagree, it is too risky."), ("C", "Fair, but the load is low."), ("B", "Okay, agreed. Let's use Redis."))
    d, _ = run(S, [0, 1, 2, 3], text="Use Redis.")
    assert d.consensus.classification == "confirmed" and d.consensus.evidence.opposition == [1] and d.consensus.score < 0.9


def test_opposition_after_agreement_is_contested():
    S = segs(("A", "We should use Redis."), ("B", "Agreed."), ("C", "Actually, I don't think that will work."))
    d, _ = run(S, [0, 1, 2], text="Use Redis.")
    assert d.consensus.classification == "contested"


OUT_NODEC = {"summary": "s", "minutes": [{"topic": "Ledger", "discussion": "Discussed.", "segment_ids": [0]}], "decisions": [], "action_items": []}


# 13 LLM2 receives consensus + reconstruction
def _trans():
    raw = {"topics": [{"topic_id": "t", "title": "Ledger", "segment_ids": [0, 1, 2, 3, 4]}], "action_candidates": [],
           "decision_candidates": [{"segment_ids": [3], "status": "confirmed", "text": "Use PostgreSQL for the ledger.",
                                    "reconstructed_text": "The team agreed to use PostgreSQL for the ledger."}]}
    res = transform_transcript(data(CONFIRMED), CFG, Fake(raw))
    return res.model_dump(mode="json")


def test_llm2_receives_consensus_and_reconstruction():
    t = _trans()
    g = render_guidance(t)
    assert "decision_1 [confirmed]" in g and "consensus: confirmed / HIGH" in g and "not a probability" in g
    assert "explicit_agreement=[2" in g and "reconstructed (model)" in g and "context segments [0, 1, 2, 3]" in g
    c = Fake(dict(OUT_NODEC, decisions=[{"decision": "Use PostgreSQL.", "segment_ids": [3]}]))
    document_transcript(data(CONFIRMED), CFG, c, transformation=t)
    system, user = c.calls[0]
    assert "consensus: confirmed / HIGH" in user and "The team agreed to use PostgreSQL" in user
    assert GUIDANCE_ADDENDUM in system and "do NOT copy it" in system and "source of truth" in system


# 14 unguided LLM2 unchanged; older transformation files still render/cross-check
def test_unguided_llm2_unchanged_and_old_transformation_compatible():
    from llm2.prompts import SYSTEM_PROMPT
    c = Fake(OUT_NODEC)
    document_transcript(data(CONFIRMED), CFG, c)
    assert c.calls[0][0] == SYSTEM_PROMPT and "consensus" not in c.calls[0][1]
    old = {"topics": [], "decision_candidates": [{"decision_id": "decision_1", "segment_ids": [3], "status": "confirmed", "text": "X."}],
           "action_candidates": []}
    assert "decision_1 [confirmed]" in render_guidance(old) and "consensus:" not in render_guidance(old)


def test_cross_check_flags_weak_consensus_without_editing():
    S = segs(("A", "I think we should adopt Kafka."), ("B", "I disagree."), ("A", "Fine, let's table it."))
    t = {"topics": [], "action_candidates": [], "decision_candidates": [dict(run(S, [0, 1, 2], status="confirmed", text="Adopt Kafka.")[0].model_dump(mode="json"))]}
    c = Fake(dict(OUT_NODEC, decisions=[{"decision": "Adopt Kafka.", "segment_ids": [0]}]))
    r = document_transcript(data(S), CFG, c, transformation=t)
    assert any("guidance consensus" in w for w in r.warnings) and len(r.decisions) == 1


# evidence / UI backend
def test_evidence_exposes_consensus_with_resolved_segments_and_old_files_still_work():
    t = _trans()
    refined = data(CONFIRMED)
    doc = {"summary": "s", "minutes": [], "decisions": [{"decision": "Use PostgreSQL.", "segment_ids": [3]}], "action_items": [], "warnings": []}
    it = build_evidence(refined, doc, t)["items"][0]
    c = it["consensus"]
    assert c["level"] == "high" and [s["id"] for s in c["evidence"]["explicit_agreement"]][:1] == [2]
    assert c["evidence"]["decision_statement"][0]["refined_text"].startswith("Should we use PostgreSQL")
    assert [s["id"] for s in c["context_segments"]] == [0, 1, 2, 3]
    assert build_evidence(refined, doc)["items"][0]["consensus"] is None          # old call signature
    assert build_evidence(refined, doc, {"decision_candidates": [{"segment_ids": [3], "status": "confirmed"}]})["items"][0]["consensus"] is None


# ---- regression: one segment confirms A and says B is still open (test5 segments 51/52/59) ----
A_TXT = "PostgreSQL remains the authoritative ledger store, while Kafka is transport/workflow coordination."
MIXED = segs(("S3", "We should record that the current architecture decision is to keep PostgreSQL as the source of truth for the ledger. "
                    "The Kafka events are transport and workflow coordination, not the authoritative balance store."),
             ("S0", "Agreed. That is the decision. The deployment strategy, final SLO, production canary date, and 30-day retention change are still open."),
             ("S3", "Okay. We have enough to write the record. The only final decision is PostgreSQL remains the authoritative ledger store, "
                    "while the rest of the rollout details remain pending."))


def test_same_segment_confirms_A_and_leaves_B_open():
    a, w = run(MIXED, [0, 1, 2], text=A_TXT, reconstructed_text="PostgreSQL remains the authoritative ledger store; Kafka events are transport and workflow coordination, not the authoritative balance store.")
    ev = a.consensus.evidence
    assert a.status == "confirmed" and a.consensus.classification == "confirmed" and a.consensus.level == "high"
    assert ev.decision_statement == [0] and ev.explicit_agreement == [1] and ev.explicit_finalization == [1, 2]
    assert ev.unresolved == [] and ev.opposition == [] and not w
    assert a.reconstruction_source == "model" and "PostgreSQL" in a.reconstructed_text
    b, _ = run(MIXED, [1], status="unresolved", text="The deployment strategy and the production canary date are still open.")
    assert b.consensus.classification == "unresolved" and b.consensus.evidence.unresolved == [1]


def test_agreed_that_is_the_decision_x_still_open_does_not_unresolve_preceding_decision():
    S = segs(("A", "We should use PostgreSQL for the ledger."), ("B", "Agreed. That is the decision. The retention period is still open."))
    d, w = run(S, [0, 1], text="Use PostgreSQL for the ledger.")
    assert d.consensus.classification == "confirmed" and d.consensus.evidence.unresolved == [] and not w
    assert cues("Agreed. That is the decision. X is still open.", "Use PostgreSQL for the ledger.")["unresolved"] is False
    assert cues("Agreed. That is the decision. X is still open.")["unresolved"] is True      # no topic: old conservative behaviour


def test_open_language_about_the_candidate_itself_still_counts():
    S = segs(("A", "We should use PostgreSQL for the ledger."), ("B", "Agreed, but using PostgreSQL for the ledger is not approved yet."))
    d, w = run(S, [0, 1], text="Use PostgreSQL for the ledger.")
    assert d.consensus.classification == "unresolved" and any("conversational evidence" in x for x in w)
    S2 = segs(("A", "We should use PostgreSQL for the ledger."), ("B", "Agreed. That is still open."))
    assert run(S2, [0, 1], text="Use PostgreSQL for the ledger.")[0].consensus.classification == "unresolved"   # anaphoric

# --- proposition-specific agreement / explicit non-approval ---
CANARY = "If the canary stays below 0.5 percent error rate for 30 minutes, increase traffic from 5 percent to 25 percent; proposed rollout policy, not approved."
ISTIO = "Istio is the service mesh currently in the staging cluster and is staying for now."


def test_A_agreement_about_other_proposition_plus_negated_approval_is_not_confirmed():
    S = segs(("A", "The rollout document proposes increasing traffic from 5 percent to 25 percent."),
             ("B", "Right. Istio is staying for now. Also, no one should infer that the proposed 25 percent traffic step has been approved just because it is in the rollout document."))
    d, _ = run(S, [0, 1], status="proposed", text=CANARY)
    ev = d.consensus.evidence
    assert d.status == "proposed" and d.consensus.classification != "confirmed" and 1 not in ev.explicit_agreement and 1 in ev.unresolved
    # the same segment still agrees about the proposition it actually restates
    S2 = segs(("A", "Istio is the service mesh we currently have in the staging cluster."), ("B", S[1]["text"]))
    i, _ = run(S2, [0, 1], text=ISTIO)
    assert i.consensus.classification == "confirmed" and i.consensus.evidence.explicit_agreement == [1]
    assert cues(S[1]["text"], CANARY)["agree"] is False and cues(S[1]["text"], ISTIO)["unresolved"] is False


def test_B_do_not_turn_recommendation_into_confirmed_decision():
    S = segs(("A", "I recommend keeping the old payment path available for 48 hours after the canary."),
             ("A", "For the documentation, please don't turn my recommendation about keeping the old path for 48 hours into a confirmed decision. It is still a recommendation."))
    d, _ = run(S, [0, 1], status="proposed", text="Keep the old payment path available for 48 hours after the canary; recommendation only, not a confirmed decision.")
    assert d.status == "proposed" and d.consensus.classification != "confirmed" and d.consensus.evidence.explicit_agreement == []


def test_C_agreed_that_is_the_decision_other_items_open_still_confirms_named_decision():
    S = segs(("A", "We should use PostgreSQL for the ledger."), ("B", "Agreed. That is the decision. Other items remain open."))
    d, w = run(S, [0, 1], text="Use PostgreSQL for the ledger.")
    assert d.consensus.classification == "confirmed" and d.consensus.evidence.explicit_agreement == [1] and d.consensus.level == "high"

# ---------------------------------------------------------------- regression: a bare "No." answering a question is not opposition
SECRETS = segs(("Speaker 04", "Can secrets be committed to the repository?"), ("Speaker 01", "No."),
               ("Speaker 01", "Secrets must not be committed to the repository."))


def test_no_answer_to_question_supports_negative_decision():
    from transformation.consensus import apply_answer_polarity
    topic = "Secrets must not be committed to the repository."
    cu = apply_answer_polarity({i: cues(s["text"], topic) for i, s in enumerate(SECRETS)}, SECRETS, topic)
    assert not cu[1]["oppose"] and cu[1]["agree"]
    d, _ = run(SECRETS, [0, 1, 2], text=topic)
    assert d.consensus.classification != "contested"
    assert d.consensus.evidence.opposition == []


def test_no_still_opposes_a_positive_decision_and_real_opposition_is_kept():
    d, _ = run(SECRETS, [0, 1], text="Secrets can be committed to the repository.")
    assert d.consensus.classification == "contested"                            # polarity does not weaken contested detection
    S = segs(("A", "Secrets must not be committed to the repository."), ("B", "I disagree."))
    assert run(S, [0, 1], text="Secrets must not be committed to the repository.")[0].consensus.classification == "contested"


# ---------------------------------------------------------------- regression: deferred / proposed / tentative is never "confirmed"
@pytest.mark.parametrize("t", ["The decision will be made later.", "We will decide this in the next meeting.", "This is only a proposal.",
                               "This is not a final decision.", "The decision has not been made yet.",
                               "The meeting is tentatively on 26 October."])
def test_deferred_or_tentative_wording_is_not_finalization(t):
    c = cues(t)
    assert not c["final"] and not c["agree"] and c["unresolved"]


@pytest.mark.parametrize("t", ["We will use PostgreSQL.", "The decision is final.", "We've decided to use Kubernetes."])
def test_genuine_finalization_still_detected(t):
    assert cues(t)["final"]


@pytest.mark.parametrize("topic,stmt", [("Adopt Kubernetes for deployment.", "We will decide this in the next meeting."),
                                        ("Set the backup retention period.", "The decision has not been made yet."),
                                        ("Hold the meeting on 26 October.", "The meeting is tentatively on 26 October.")])
def test_deferred_decisions_are_not_classified_confirmed(topic, stmt):
    S = segs(("A", topic), ("B", stmt))
    assert run(S, [0, 1], status="proposed", text=topic)[0].consensus.classification in ("unresolved", "proposed", "contested")
