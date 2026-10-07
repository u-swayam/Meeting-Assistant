"""Decision context reconstruction: a self-contained, GROUNDED decision statement (Inoue et al. inspired).

The model MAY propose `reconstructed_text`. It is accepted only if it passes deterministic grounding checks against the
selected context (numbers/identifiers, negation, hedging/status, vocabulary). Otherwise a deterministic fallback built
ONLY from existing text is used. Nothing here can add a fact that is not in the cited context.
"""
from __future__ import annotations
import re

from transformation.consensus import _norm, is_context_dependent

_FILLER = re.compile(r"^(?:(?:yeah|yes|yep|okay|ok|well|so|um|uh|right|alright|all right|sure|then|and|but)[\s,.!-]+)+", re.I)
_TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./\-]*")
_NEG = re.compile(r"\b(not|never|no|cannot|without|neither|nor|none)\b|n't\b")
_STATUS_PHRASE = re.compile(r"\bnot (yet )?(been )?(decided|agreed|approved|confirmed|final|settled)\b|\bno (agreement|decision|consensus)\b"
                            r"|\bnot (agreed|approved)\b", re.I)
_DEFINITE = re.compile(r"\b(agreed|decided|confirmed|finalized|finalised|approved|settled)\b")
_STATUS_MARK = re.compile(r"propos|suggest|tentativ|unresolved|\bopen\b|discuss|consider|might|could|should|maybe|probably|"
                          r"reject|disagree|oppos|contest|not (yet )?(decided|agreed|approved|confirmed)|no (agreement|decision)", re.I)
_ALLOW = set("""the and for that this with from into will would should could might shall team group meeting participants participant
speaker speakers agreed agree decided decision decisions proposed proposal propose suggested suggest suggestion confirmed
finalized finalised approved tentative tentatively unresolved open opposition opposed disagreed contested discussed discussion
use used using yet remains remain still not that there their which were been being have has had also only
instead rather about over under after before when then than they them was are but""".split())


def clean_filler(text: str) -> str:
    t = _FILLER.sub("", text.strip()).strip()
    return (t[:1].upper() + t[1:]) if t else text.strip()


def _toks(s: str) -> list[str]:
    return [t.lower().rstrip(".-/") for t in _TOKEN.findall(s)]


def grounding_issue(candidate: str, ctx_text: str, cited_text: str, classification: str) -> str | None:
    """None if `candidate` is grounded in the context, else a short reason."""
    cand, ctx, cited = _norm(candidate), _norm(ctx_text), _norm(cited_text)
    if not cand:
        return "empty"
    ctx_toks = set(_toks(ctx_text))
    full = ctx_toks
    pref = {t[:4] for t in ctx_toks if len(t) >= 4}
    for t in _toks(candidate):
        if not t:
            continue
        ident = any(ch.isdigit() for ch in t) or any(ch in t for ch in "_/.-")
        if ident:
            if t not in ctx:
                return f"number/identifier {t!r} not in context"
            continue
        if len(t) < 4 or t in _ALLOW:
            continue
        if t not in full and t[:4] not in pref:
            return f"term {t!r} not found in context"
    stripped = _STATUS_PHRASE.sub(" ", cand)
    if _NEG.search(cited) and not _NEG.search(stripped):
        return "negation present in the decision segments was lost"
    if _NEG.search(stripped) and not _NEG.search(ctx):
        return "negation not present in the context was introduced"
    if classification != "confirmed":
        if _DEFINITE.search(_STATUS_PHRASE.sub(" ", cand)):
            return "presents a non-confirmed decision as agreed/decided"
        if not _STATUS_MARK.search(cand):
            return "uncertain/unconfirmed status not preserved"
    return None


def status_label(status: str, classification: str) -> str:
    if status == "confirmed":
        return "Confirmed" if classification == "confirmed" else f"Candidate marked confirmed, but consensus evidence is '{classification}'"
    return {"tentative": "Tentative", "proposed": "Proposed (agreement not established)",
            "unresolved": "Unresolved (not decided)"}.get(status, status.capitalize())


def reconstruct(model_text, candidate_text: str, status: str, classification: str, segments: list[dict],
                ctx: list[int], statement: int, trigger: int, cited: list[int]):
    """Return (reconstructed_text | None, source, warning | None)."""
    if not ctx:
        return None, "none", None
    ctx_text = " ".join(segments[i]["text"] for i in ctx)
    cited_text = " ".join(segments[i]["text"] for i in cited)
    warn = None
    if isinstance(model_text, str) and model_text.strip():
        why = grounding_issue(model_text, ctx_text, cited_text, classification)
        if why is None:
            return model_text.strip(), "model", None
        warn = f"model reconstructed_text rejected ({why}); deterministic fallback used"
    label = status_label(status, classification)
    if candidate_text and grounding_issue(candidate_text, ctx_text, cited_text, "confirmed" if status == "confirmed" else classification) is None:
        return f"{label}: {candidate_text.strip()}", "candidate_text", warn
    src = statement if is_context_dependent(segments[trigger]["text"]) or trigger == statement else trigger
    body = clean_filler(segments[src]["text"])
    return (f"{label}: {body}" if body else None), "original_text", warn
