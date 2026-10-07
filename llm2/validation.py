"""Conservative deterministic validation of LLM2 output.

Structural problems (missing/mistyped required fields) raise LLM2OutputError so the caller can
retry once. Item-level problems drop only that item with a warning. Grounding checks that
cannot be decided deterministically are WARNINGS only (flags for human review), never silent edits.
"""
from __future__ import annotations
import re

from pydantic import ValidationError

from llm2.errors import LLM2OutputError
from llm2.schemas import ActionItem, DecisionItem, MinuteItem

_NO_VALUE = {"", "null", "none", "n/a", "na", "unknown", "unassigned", "tbd", "not stated",
             "not specified", "not provided", "unspecified"}
_SPEAKER_LABEL = re.compile(r"^speaker[_ ]?\d+$", re.I)
_NUM = re.compile(r"\d[\d.,]*\d|\d")
_TENTATIVE = re.compile(r"\b(could|might|may|maybe|perhaps|possibly|suggest\w*|propos\w*|recommend\w*|"
                        r"tentative\w*|pending|consider\w*)\b", re.I)
_UNRESOLVED = re.compile(r"(not (yet )?(been )?(approved|decided|agreed|final)|hasn't been (approved|decided)|"
                         r"haven't (decided|agreed)|still (open|pending|undecided)|\bpending\b|\bunresolved\b)", re.I)
_CONDITIONAL = re.compile(r"\b(if|might|maybe|probably|hopefully|tentative\w*|unless)\b", re.I)


def _clean_optional(v, what: str, warns: list[str], where: str):
    """owner/deadline: str or None; blank / 'unknown'-style placeholders become None."""
    if v is None:
        return None
    if not isinstance(v, str):
        warns.append(f"{where}: {what} was not a string; set to null")
        return None
    t = v.strip()
    if t.lower() in _NO_VALUE:
        return None
    return t


def _ids(raw_ids, n: int, where: str, warns: list[str]) -> list[int]:
    out = []
    for x in raw_ids if isinstance(raw_ids, list) else []:
        if isinstance(x, int) and not isinstance(x, bool) and 0 <= x < n:
            if x not in out:
                out.append(x)
        else:
            warns.append(f"{where}: invalid segment id {x!r} removed")
    return sorted(out)


def _build(model, d, n: int, w: str, warns: list[str]):
    """Validate an item; segment_ids are cleaned separately so one bad id doesn't drop the item."""
    if not isinstance(d, dict):
        return None
    raw_ids = d.get("segment_ids")
    try:
        item = model.model_validate({k: v for k, v in d.items() if k != "segment_ids"})
    except ValidationError:
        return None
    item.segment_ids = _ids(raw_ids, n, w, warns)
    return item


def _numbers_missing(text: str, evidence: str) -> list[str]:
    return [m for m in dict.fromkeys(_NUM.findall(text)) if m.strip(".,") not in evidence]


def validate_response(raw: dict, segments: list[dict]) -> tuple[dict, list[str]]:
    """Return ({'summary','minutes','decisions','action_items'}, warnings) or raise LLM2OutputError."""
    if not isinstance(raw, dict) or "__invalid_provider_output__" in raw:
        why = raw.get("__invalid_provider_output__") if isinstance(raw, dict) else "not an object"
        raise LLM2OutputError(f"model output unusable: {why}")
    summary = raw.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise LLM2OutputError("missing or empty 'summary'")
    for k in ("minutes", "decisions", "action_items"):
        if not isinstance(raw.get(k), list):
            raise LLM2OutputError(f"'{k}' missing or not a list")

    n = len(segments)
    text_of = lambda ids: " ".join(segments[i]["text"] for i in ids).lower()
    all_text = " ".join(s["text"] for s in segments).lower()
    warns: list[str] = []

    minutes = []
    for k, m in enumerate(raw["minutes"]):
        w = f"minutes[{k}]"
        item = _build(MinuteItem, m, n, w, warns)
        if item is None:
            warns.append(f"{w}: malformed; dropped"); continue
        if not item.topic.strip() or not item.discussion.strip():
            warns.append(f"{w}: empty topic/discussion; dropped"); continue
        if not item.segment_ids:
            warns.append(f"{w}: no valid segment_ids (kept; unverifiable)")
        for num in _numbers_missing(item.discussion, text_of(item.segment_ids) if item.segment_ids else all_text):
            warns.append(f"{w}: number '{num}' not found in cited segments")
        minutes.append(item)
    if not minutes:
        raise LLM2OutputError("no valid minutes items")

    decisions = []
    for k, d in enumerate(raw["decisions"]):
        w = f"decisions[{k}]"
        item = _build(DecisionItem, d, n, w, warns)
        if item is None:
            warns.append(f"{w}: malformed; dropped"); continue
        if not item.decision.strip():
            warns.append(f"{w}: empty; dropped"); continue
        if not item.segment_ids:
            warns.append(f"{w}: no valid segment_ids (unsupported by evidence); dropped"); continue
        ev = text_of(item.segment_ids)
        if _TENTATIVE.search(item.decision):
            warns.append(f"{w}: wording is tentative/proposal-like; verify it was actually agreed")
        if _UNRESOLVED.search(ev):
            warns.append(f"{w}: cited segments mention pending/unapproved status; verify")
        for num in _numbers_missing(item.decision, ev):
            warns.append(f"{w}: number '{num}' not found in cited segments")
        decisions.append(item)

    actions = []
    for k, a in enumerate(raw["action_items"]):
        w = f"action_items[{k}]"
        if not isinstance(a, dict):
            warns.append(f"{w}: malformed; dropped"); continue
        a = dict(a)
        a["owner"] = _clean_optional(a.get("owner"), "owner", warns, w)
        a["deadline"] = _clean_optional(a.get("deadline"), "deadline", warns, w)
        item = _build(ActionItem, a, n, w, warns)
        if item is None:
            warns.append(f"{w}: malformed; dropped"); continue
        if not item.task.strip():
            warns.append(f"{w}: empty task; dropped"); continue
        if not item.segment_ids:
            warns.append(f"{w}: no valid segment_ids (unsupported by evidence); dropped"); continue
        ev = text_of(item.segment_ids)
        if item.owner:
            if _SPEAKER_LABEL.match(item.owner):
                warns.append(f"{w}: owner '{item.owner}' is a speaker label (inferred); set to null")
                item.owner = None
            elif not any(tok in all_text for tok in re.findall(r"\w{2,}", item.owner.lower())):
                warns.append(f"{w}: owner '{item.owner}' never appears in the transcript; set to null")
                item.owner = None
        if item.deadline:
            if _CONDITIONAL.search(ev):
                warns.append(f"{w}: deadline '{item.deadline}' cited near conditional/tentative wording; verify")
            for num in _numbers_missing(item.deadline, all_text):
                warns.append(f"{w}: deadline number '{num}' not found in the transcript; verify")
        for num in _numbers_missing(item.task, ev):
            warns.append(f"{w}: number '{num}' not found in cited segments")
        actions.append(item)

    return {"summary": summary.strip(), "minutes": minutes, "decisions": decisions,
            "action_items": actions}, warns
