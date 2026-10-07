"""Deterministic consensus analysis + decision-context selection/reconstruction (CFAS / Inoue et al. inspired).

PURE functions over the refined transcript: no model call, no I/O, nothing invented.
  * Cues are lexical, per segment (agreement, finalization, opposition, unresolved, proposal/hedge, question).
  * Consensus is derived from WHICH segments carry those cues and in WHAT ORDER; every classification and score is
    traceable to segment ids (`evidence`). Silence is never agreement.
  * The score is a HEURISTIC internal confidence, not a calibrated probability (`score_kind` says so).
  * Consensus is an ADDITIONAL layer: it never rewrites a candidate's existing `status`.
"""
from __future__ import annotations
import re

SCORE_KIND = "heuristic_internal_confidence_not_calibrated"


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.replace("\u2019", "'").replace("\u2018", "'").lower()).strip()


def _rx(*pats: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in pats))


_OPPOSE = _rx(r"\bi (strongly )?disagree\b", r"\bwe disagree\b", r"\bi don'?t (think|agree|like)\b[^.?!]{0,40}",
              r"\bi do not (think|agree)\b", r"\blet'?s not\b", r"\bi'?m (against|opposed)\b", r"\bi object\b",
              r"\b(that|this|it) (won'?t|will not|wouldn'?t|doesn'?t|isn'?t going to) work\b", r"\bnot a good idea\b",
              r"\bi'?d rather not\b", r"\bwe shouldn'?t\b", r"\bi would not\b", r"\bi can'?t agree\b",
              r"\bwe (can'?t|cannot) (do|use|go with)\b", r"^no\b[,.!]?", r"\bthat'?s a no\b",
              r"\b(reject|rejected|veto)\b")
_UNRES = _rx(r"\b(not|n't|never|hasn'?t|haven'?t)\b[^.?!]{0,20}\b(yet )?(been )?(decided|agreed|approved|final|settled|confirmed)\b",
             r"\bhaven'?t (decided|agreed)\b", r"\bundecided\b", r"\bunresolved\b", r"\bpending\b",
             r"\bstill (open|undecided|to be decided)\b", r"\b(open|outstanding) (question|issue|item)\b",
             r"\b(discuss|revisit|decide|come back to) (it |this |that )?(later|next (week|time|meeting)|again)\b",
             r"\blet'?s (discuss|revisit|table|park|defer)\b", r"\b(table|defer|postpone)(d)? (this|it|that)\b",
             r"\bto be decided\b", r"\bnot approved\b", r"\bleave (it|this) open\b", r"^maybe\b[.,]?\s*$", r"^maybe[.,]")
_HEDGE = _rx(r"\bi think\b", r"\bmaybe\b", r"\bperhaps\b", r"\bwhat if\b", r"\bwe (could|might|may)\b", r"\bcould we\b",
             r"\bshould we\b", r"\bhow about\b", r"\bi (suggest|propose|recommend)\b", r"\bprobably\b", r"\bwe should\b",
             r"\bwe ought to\b", r"\bit might\b", r"\bpossibly\b", r"\bi'?d (suggest|propose|go with)\b", r"\bwould it make sense\b")
_FINAL = _rx(r"\bwe'?ll (use|go with|adopt|take|do)\b", r"\bwe will (use|go with|adopt|take|do)\b",
             r"\bwe'?re going (with|to use)\b", r"\bwe are going (with|to use)\b", r"\blet'?s (go with|use|do|make)\b",
             r"\b(the|our|that'?s the|that is the|that'?s our|that is our)( final)? decision( is)?\b", r"\bfinal (decision|approach|answer)\b",
             r"\bwe'?ve (decided|agreed|settled)\b", r"\bwe (decided|agreed|settled) (on|to)\b", r"\bdecided to\b",
             r"\bthat'?s (settled|final)\b", r"\bit'?s decided\b", r"\bthat is the decision\b")
_AGREE = _rx(r"\bagreed\b", r"\bi agree\b", r"\bwe agree\b", r"\bthat works\b", r"\bworks for (me|us)\b", r"\bi'?m on board\b",
             r"\b(yes|yeah|yep|sure|okay|ok|alright|all right),? (let'?s|we'?ll|we will|do that|go for it|go ahead)\b",
             r"\blet'?s do (that|it|this)\b", r"\bgo(ing)? with (that|this|it)\b", r"\bi'?m (fine|good|ok|okay) with (that|this|it)\b",
             r"\bapproved\b", r"\bsigned off\b", r"\bconfirmed\b", r"\bfine by me\b", r"^(yes|yeah|yep)[.!]?$", r"^(okay|ok)[,.!]? (let'?s|we'?ll)")
_WEAK = _rx(r"\bsounds (good|great|fine|reasonable)\b", r"\bmakes sense\b", r"\bi like (it|that|this)\b", r"\bgood (idea|point)\b",
            r"\bi think so\b", r"\bprobably (fine|right|works)\b", r"\bseems (good|fine|reasonable|right)\b", r"\bnot opposed\b",
            r"\bi'?m open to\b", r"\bmaybe\b[^.?!]{0,10}\b(works|good|fine)\b", r"^(good|great|nice|right|true)[.!]?$")
# Explicit refusal to treat a proposition as approved/decided ("no one should infer X has been approved", "don't turn my
# recommendation into a confirmed decision", "it is still a recommendation"). Proposition-specific: see cues().
_NONAPPROVAL = _rx(
    r"\b(no ?one|nobody|don'?t|do not|please don'?t|should(n'?t| not)|must(n'?t| not)|can'?t|cannot)\b[^.?!]{0,80}"
    r"\b(infer|assume|treat|turn|read|interpret|take|record|document|write|mark|present|consider|call|make)\b[^.?!]{0,120}"
    r"\b(approved|confirmed|decided|decision|final|agreed|settled)\b",
    r"\bstill (just )?(a |an )?(recommendation|proposal|proposed|suggestion)\b",
    r"\b(only|just|merely) (a |an )?(recommendation|proposal|suggestion)\b",
    r"\b(recommendation|proposal|suggestion) (only|not a (confirmed|final|approved)? ?decision)\b")
_AFFIRM = re.compile(r"^(right|yes|yeah|yep|exactly|correct|indeed)[.!,]?$")
_NEG_BEFORE_AGREE = re.compile(r"\b(not|n't|never|no|without|dis)\s*$")
_ANAPHORA = re.compile(r"\b(that|it|this|those|them|the same|so)\b")


_SPLIT = re.compile(r"(?<=[.!?;])\s+|,?\s+(?=(?:while|whereas|but|although|though)\b)", re.I)
_GENERIC = set("""that this with from have been were will should would could still open pending final decision decisions decided approved
agreed agree remain remains rest only unresolved yet next later week there their which about also then than they them are but
not has had and the for while whereas although though because also into onto over""".split())


def _content(t: str) -> set:
    return {w[:4] for w in re.findall(r"[a-z0-9][a-z0-9_\-]*", t) if len(w) >= 4 and w not in _GENERIC}


def clauses(text: str) -> list[str]:
    return [c for c in (x.strip() for x in _SPLIT.split(text)) if c] or [text]


def relevant(clause: str, topic: str | None) -> bool:
    """Does an open/pending/opposing clause concern THIS candidate? No topic -> always (old behaviour). A clause with no
    content words of its own ("That is still open.") is anaphoric and refers to the candidate."""
    if not topic:
        return True
    c = _content(_norm(clause))
    if not c:                      # no content words: only a pronoun/demonstrative ("That is still open") points back at the candidate
        return bool(re.search(r"\b(that|this|it|they|those|these|the (decision|proposal|plan|choice))\b", _norm(clause)))
    return bool(c & _content(_norm(topic)))


def _clause_cues(t: str) -> dict:
    c = {"oppose": bool(_OPPOSE.search(t)), "unresolved": bool(_UNRES.search(t)), "hedge": bool(_HEDGE.search(t)),
         "question": t.rstrip().endswith("?"), "final": False, "agree": False, "weak": False,
         "nonapproval": bool(_NONAPPROVAL.search(t))}
    c["unresolved"] = c["unresolved"] or c["nonapproval"]
    if not (c["oppose"] or c["unresolved"] or c["question"]):
        agree = any(not _NEG_BEFORE_AGREE.search(t[:m.start()]) for m in _AGREE.finditer(t))
        final = any(not _NEG_BEFORE_AGREE.search(t[:m.start()]) for m in _FINAL.finditer(t))
        c["final"] = final and not c["hedge"]
        c["agree"] = agree and not c["hedge"]
        c["weak"] = bool(_WEAK.search(t)) and not (c["agree"] or c["final"]) and not c["hedge"]
    return c


def cues(text: str, topic: str | None = None) -> dict:
    """Lexical cues for ONE segment, evaluated PER CLAUSE. `topic` (the candidate's text) makes it proposition-specific:
    opposition/unresolved clauses about OTHER matters ("The SLO is still open.") do not count against this candidate,
    while ones about the candidate (or anaphoric ones) still do. Agreement/finalization is never counted from a clause
    that is itself hedged/questioned/opposed, nor when a relevant opposition/unresolved clause is in the same segment."""
    cl = [_clause_cues(_norm(c)) for c in clauses(text)]
    raw = clauses(text)
    # Only a MIXED segment (it also contains a clean agreement/finalization clause) is split by proposition; a segment with
    # no such clause keeps the conservative behaviour (every open/opposing clause counts).
    mixed = any(c["final"] or c["agree"] for c in cl) or (bool(topic) and any(c["nonapproval"] for c in cl))
    rel = (lambda r: relevant(r, topic)) if mixed else (lambda r: True)
    neg_o = any(c["oppose"] and rel(r) for c, r in zip(cl, raw))
    neg_u = any(c["unresolved"] and rel(r) for c, r in zip(cl, raw))
    out = {"oppose": neg_o, "unresolved": neg_u, "hedge": any(c["hedge"] for c in cl),
           "question": any(c["question"] for c in cl), "final": False, "agree": False, "weak": False}
    # In a segment that also talks about ANOTHER matter, agreement/finalization counts only from clauses that are about
    # this candidate (or contentless like "Agreed."): "The only final decision is X" does not finalize candidate Y.
    other_matter = bool(topic) and mixed is not None and any((c["oppose"] or c["unresolved"]) and not relevant(r, topic)
                                                             for c, r in zip(cl, raw))
    if other_matter or (topic and not mixed and any((c["oppose"] or c["unresolved"]) and not relevant(r, topic)
                                                    for c, r in zip(cl, raw))):
        pos = [c for c, r in zip(cl, raw) if not _content(_norm(r)) or _content(_norm(r)) & _content(_norm(topic))]
    else:
        pos = cl
    if not (neg_o or neg_u):
        out["final"] = any(c["final"] for c in pos)
        out["agree"] = any(c["agree"] for c in pos)
        out["weak"] = any(c["weak"] for c in pos) and not (out["agree"] or out["final"])
    # A bare affirmation ("Right.", "Exactly.") agrees with the proposition the SAME segment goes on to restate, so it counts as
    # agreement for a candidate only when another clause of that segment is about the candidate and is not itself negated/hedged.
    if topic and not (out["oppose"] or out["unresolved"] or out["agree"] or out["final"]) and len(raw) > 1 \
            and _AFFIRM.match(_norm(raw[0])):
        out["agree"] = any(relevant(r, topic) and _content(_norm(r)) and not (c["oppose"] or c["unresolved"] or c["hedge"]
                                                                          or c["question"])
                           for c, r in list(zip(cl, raw))[1:])
        out["weak"] = out["weak"] and not out["agree"]
    return out


def _statement_of(cited, cu):
    for i in cited:
        if (cu[i]["final"] or cu[i]["hedge"]) and not cu[i]["oppose"]:
            return i
    for i in cited:
        if not (cu[i]["agree"] or cu[i]["weak"] or cu[i]["oppose"] or cu[i]["unresolved"]):
            return i
    return cited[0]


def is_context_dependent(text: str) -> bool:
    """Inoue et al.: short anaphoric / agreeing utterances are not understandable alone."""
    t = _norm(text)
    return len(t.split()) <= 10 and (bool(_ANAPHORA.search(t)) or cues(text)["agree"])


def select_context(cited: list[int], segments: list[dict], cu: dict | None = None, max_total: int = 8) -> list[int]:
    """Bounded context: cited segments + preceding segments that introduce the issue (more when the decision
    utterance is context-dependent) + immediately following segments ONLY while they carry
    agreement/opposition/unresolved language. Stops at a new proposal. All ids are valid positions."""
    n = len(segments)
    cited = sorted({i for i in cited if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n})
    if not cited:
        return []
    cu = cu or {i: cues(s["text"]) for i, s in enumerate(segments)}
    ctx = set(cited)
    first, last = cited[0], cited[-1]
    limit = 3 if is_context_dependent(segments[first]["text"]) else 1
    i = first - 1
    while i >= 0 and limit > 0 and len(ctx) < max_total:
        if i not in ctx:
            ctx.add(i)
        limit -= 1
        i -= 1
    j, steps = last + 1, 0
    while j < n and steps < 3 and len(ctx) < max_total:
        c = cu[j]
        if not (c["agree"] or c["weak"] or c["oppose"] or c["unresolved"] or c["final"]) or (c["hedge"] and not c["oppose"]):
            break
        ctx.add(j)
        j += 1
        steps += 1
    return sorted(ctx)


def pick_trigger(ctx: list[int], statement: int, cu: dict) -> int:
    for key in ("final", "agree"):
        hit = [i for i in ctx if cu[i][key]]
        if hit:
            return hit[-1]
    return statement


def analyze_consensus(cited: list[int], ctx: list[int], segments: list[dict], cu: dict | None = None) -> dict:
    """Return a Consensus dict. Evidence lists are segment ids only; opposition/unresolved are never dropped."""
    cu = cu or {i: cues(s["text"]) for i, s in enumerate(segments)}
    cited = sorted(cited)
    ids = sorted(set(ctx) | set(cited))
    stmt = _statement_of(cited, cu)
    if is_context_dependent(segments[stmt]["text"]):      # "Okay, let's go with that." -> the proposal it refers to
        prev = [i for i in ctx if i < stmt and (cu[i]["question"] or cu[i]["hedge"] or cu[i]["final"])
                and not is_context_dependent(segments[i]["text"])]
        stmt = prev[-1] if prev else stmt
    spk = lambda i: segments[i]["speaker"]
    after = [i for i in ids if i > stmt]
    agree = [i for i in after if cu[i]["agree"]]
    final = [i for i in ids if i >= stmt and cu[i]["final"]]
    weak = [i for i in after if cu[i]["weak"]]
    oppose = [i for i in after if cu[i]["oppose"]]
    unres = [i for i in ids if i >= stmt and cu[i]["unresolved"]]
    supporting = [i for i in after if i in cited and i not in agree + weak + oppose + unres + final
                  and not cu[i]["question"] and spk(i) != spk(stmt)]
    supporting = sorted(set(weak + supporting))
    strong = sorted(set(agree + final))
    neg = sorted(set(oppose + unres))
    notes: list[str] = []
    others = {spk(i) for i in agree + weak + supporting if spk(i) != spk(stmt)}
    if strong and (not neg or strong[-1] > neg[-1]):
        cls = "confirmed"
        score = 0.55
        if any(spk(i) != spk(stmt) for i in agree):
            score += 0.20
        if final:
            score += 0.10
        if len(others) >= 2:
            score += 0.10
        if supporting:
            score += 0.05
        if neg:
            score -= 0.20
            notes.append("opposition/unresolved language earlier in the context was followed by explicit agreement")
        if not agree and final and not others:
            notes.append("explicit finalization wording without a separate agreeing response")
    elif neg:
        cls = "unresolved" if (unres and unres[-1] >= (oppose[-1] if oppose else -1)) else "contested"
        score = 0.10
        notes.append("the last decisive utterance is " + ("an explicit leave-open/unresolved statement" if cls == "unresolved"
                                                       else "opposition/disagreement"))
    elif supporting:
        cls, score = "likely", 0.35 + 0.10 * min(2, len(others))
        notes.append("supporting response(s) exist but no explicit agreement or finalization")
    else:
        cls, score = "proposed", 0.15
        notes.append("no agreement, finalization or opposition found; silence is not treated as agreement")
    score = round(max(0.0, min(0.97, score)), 2)
    level = "high" if score >= 0.75 else "medium" if score >= 0.45 else "low"
    return {"classification": cls, "level": level, "score": score, "score_kind": SCORE_KIND,
            "evidence": {"decision_statement": [stmt], "supporting_statements": supporting,
                         "explicit_agreement": agree, "explicit_finalization": final,
                         "opposition": oppose, "unresolved": unres},
            "notes": notes}