"""Strict deterministic validation of the Transformation model output.

Structural problems raise TransformationError (caller retries once, then fails clearly).
Item-level problems drop only that item with a warning. Owners/deadlines are never trusted blindly."""
from __future__ import annotations
import re
import typing

from pydantic import ValidationError

from transformation.consensus import analyze_consensus, cues, pick_trigger, select_context
from transformation.errors import TransformationError
from transformation.reconstruction import reconstruct
from transformation.schemas import (ActionCandidate, ActionStatus, DecisionCandidate, DecisionStatus,
                                    DiscussionUnit, Relation, Topic, UnitType)

_NO_VALUE = {"", "null", "none", "n/a", "na", "unknown", "unassigned", "tbd", "not stated", "not specified"}
_LABEL = re.compile(r"^speaker[_ ]?\d+$", re.I)
_UNRESOLVED = re.compile(r"(not (yet )?(been )?(approved|decided|agreed|final)|haven't (decided|agreed)|"
                         r"hasn't been (approved|decided)|\bpending\b|\bunresolved\b|still (open|undecided))", re.I)
_UNITS = set(typing.get_args(UnitType))


def _opt(v, what, w, warns):
    if v is None:
        return None
    if not isinstance(v, str):
        warns.append(f"{w}: {what} was not a string; set to null")
        return None
    t = v.strip()
    return None if t.lower() in _NO_VALUE else t


def _ids(raw, n, w, warns):
    out = []
    for x in raw if isinstance(raw, list) else []:
        if isinstance(x, int) and not isinstance(x, bool) and 0 <= x < n:
            if x not in out:
                out.append(x)
        else:
            warns.append(f"{w}: invalid segment id {x!r} removed")
    return sorted(out)


def validate_response(raw, segments: list[dict]):
    """Return (dict(topics, discussion_units, decision_candidates, action_candidates, relations), warnings)."""
    if not isinstance(raw, dict) or "__invalid_provider_output__" in raw:
        why = raw.get("__invalid_provider_output__") if isinstance(raw, dict) else "not an object"
        raise TransformationError(f"model output unusable: {why}")
    for k in ("topics", "decision_candidates", "action_candidates"):
        if not isinstance(raw.get(k), list):
            raise TransformationError(f"'{k}' missing or not a list")
    n = len(segments)
    all_text = " ".join(s["text"] for s in segments).lower()
    text_of = lambda ids: " ".join(segments[i]["text"] for i in ids)
    warns: list[str] = []

    # ---- topics (re-id deterministically in meeting order; remap references) ----
    topics, remap = [], {}
    for k, t in enumerate(raw["topics"]):
        w = f"topics[{k}]"
        if not isinstance(t, dict) or not isinstance(t.get("title"), str) or not t["title"].strip():
            warns.append(f"{w}: malformed/empty title; dropped"); continue
        ids = _ids(t.get("segment_ids"), n, w, warns)
        if not ids:
            warns.append(f"{w}: no valid segment_ids; dropped"); continue
        topics.append((ids, t["title"].strip(), t.get("topic_id")))
    if not topics:
        raise TransformationError("no valid topics")
    topics.sort(key=lambda x: x[0][0])
    out_topics = []
    for j, (ids, title, old) in enumerate(topics, 1):
        tid = f"topic_{j}"
        if isinstance(old, str):
            remap.setdefault(old, tid)
        out_topics.append(Topic(topic_id=tid, title=title, segment_ids=ids))
    topic_ids = {t.topic_id for t in out_topics}

    def topic_ref(v, w):
        if v is None:
            return None
        t = remap.get(v) if isinstance(v, str) else None
        if t is None:
            warns.append(f"{w}: unknown topic_id {v!r}; set to null")
        return t

    units = []
    for k, u in enumerate(raw.get("discussion_units") or []):
        w = f"discussion_units[{k}]"
        if not isinstance(u, dict):
            warns.append(f"{w}: malformed; dropped"); continue
        ids = _ids(u.get("segment_ids"), n, w, warns)
        if not ids:
            warns.append(f"{w}: no valid segment_ids; dropped"); continue
        typ = u.get("type")
        if typ not in _UNITS:
            warns.append(f"{w}: unknown type {typ!r}; set to 'discussion'"); typ = "discussion"
        units.append(DiscussionUnit(discussion_id=f"discussion_{len(units) + 1}",
                                    topic_id=topic_ref(u.get("topic_id"), w), segment_ids=ids, type=typ))

    decisions = []
    for k, d in enumerate(raw["decision_candidates"]):
        w = f"decision_candidates[{k}]"
        if not isinstance(d, dict) or not isinstance(d.get("text"), str) or not d["text"].strip():
            warns.append(f"{w}: malformed/empty; dropped"); continue
        if d.get("status") not in typing.get_args(DecisionStatus):
            warns.append(f"{w}: invalid status {d.get('status')!r}; dropped"); continue
        ids = _ids(d.get("segment_ids"), n, w, warns)
        if not ids:
            warns.append(f"{w}: no valid segment_ids (unsupported by evidence); dropped"); continue
        # --- consensus + context reconstruction (deterministic; status is never modified) ---
        cue_map = {i: cues(segments[i]["text"], d["text"]) for i in range(n)}      # candidate-specific cues
        ctx = select_context(ids, segments, cue_map)
        cons = analyze_consensus(ids, ctx, segments, cue_map)
        stmt = cons["evidence"]["decision_statement"][0]
        trig = pick_trigger(ctx, stmt, cue_map)
        rtext, rsrc, rwarn = reconstruct(d.get("reconstructed_text"), d["text"].strip(), d["status"],
                                         cons["classification"], segments, ctx, stmt, trig, ids)
        if d["status"] == "confirmed" and cons["classification"] != "confirmed" and _UNRESOLVED.search(text_of(ids)):
            warns.append(f"{w}: marked confirmed but cited segments mention pending/unapproved status; verify")
        if rwarn:
            warns.append(f"{w}: {rwarn}")
        if d["status"] == "confirmed" and cons["classification"] != "confirmed":
            warns.append(f"{w}: marked confirmed but conversational evidence is '{cons['classification']}' "
                         f"(consensus {cons['level']}); verify it was agreed")
        decisions.append(DecisionCandidate(decision_id=f"decision_{len(decisions) + 1}",
                                           topic_id=topic_ref(d.get("topic_id"), w), segment_ids=ids,
                                           status=d["status"], text=d["text"].strip(), consensus=cons,
                                           trigger_segment=trig, context_segments=ctx,
                                           original_text=segments[trig]["text"], reconstructed_text=rtext,
                                           reconstruction_source=rsrc))

    actions = []
    for k, a in enumerate(raw["action_candidates"]):
        w = f"action_candidates[{k}]"
        if not isinstance(a, dict) or not isinstance(a.get("description"), str) or not a["description"].strip():
            warns.append(f"{w}: malformed/empty; dropped"); continue
        if a.get("status") not in typing.get_args(ActionStatus):
            warns.append(f"{w}: invalid status {a.get('status')!r}; dropped"); continue
        ids = _ids(a.get("segment_ids"), n, w, warns)
        if not ids:
            warns.append(f"{w}: no valid segment_ids (unsupported by evidence); dropped"); continue
        owner = _opt(a.get("owner"), "owner", w, warns)
        deadline = _opt(a.get("deadline"), "deadline", w, warns)
        cond = _opt(a.get("condition"), "condition", w, warns)
        if owner and _LABEL.match(owner):
            warns.append(f"{w}: owner '{owner}' is a speaker label (inferred); set to null"); owner = None
        elif owner and not any(tok in all_text for tok in re.findall(r"\w{2,}", owner.lower())):
            warns.append(f"{w}: owner '{owner}' never appears in the transcript; set to null"); owner = None
        if a["status"] == "conditional" and not cond:
            warns.append(f"{w}: conditional action without a stated condition; verify")
        actions.append(ActionCandidate(action_id=f"action_{len(actions) + 1}", topic_id=topic_ref(a.get("topic_id"), w),
                                       segment_ids=ids, status=a["status"], description=a["description"].strip(),
                                       owner=owner, deadline=deadline, condition=cond))

    rels = []
    for k, r in enumerate(raw.get("relations") or []):
        try:
            rel = Relation.model_validate(r)
        except ValidationError:
            warns.append(f"relations[{k}]: malformed; dropped"); continue
        if not (0 <= rel.from_segment < n and 0 <= rel.to_segment < n):
            warns.append(f"relations[{k}]: invalid segment id; dropped"); continue
        rels.append(rel)

    covered = {i for t in out_topics for i in t.segment_ids}
    if len(covered) < n:
        warns.append(f"{n - len(covered)} of {n} segments are not assigned to any topic")
    return {"topics": out_topics, "discussion_units": units, "decision_candidates": decisions,
            "action_candidates": actions, "relations": rels}, warns