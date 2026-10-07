"""Deterministic guards applied to the model's output.

The model never controls speaker/time/original_text: those are copied from
the input. The audit trail is derived from a word-level diff of original vs
refined text, so every modification is documented. Risky or low-confidence
spans are reverted.
"""

from __future__ import annotations

import difflib
import re

from llm1.schemas import Change, ModelSegment


_NEGATIONS = {
    "not",
    "no",
    "never",
    "none",
    "nothing",
    "nobody",
    "neither",
    "nor",
    "cannot",
    "without",
    "can't",
    "couldn't",
    "didn't",
    "doesn't",
    "don't",
    "isn't",
    "wasn't",
    "won't",
    "wouldn't",
    "shouldn't",
    "mustn't",
}

_HEDGES_COMMITMENTS = {
    "might",
    "may",
    "maybe",
    "perhaps",
    "probably",
    "possibly",
    "think",
    "guess",
    "suppose",
    "should",
    "would",
    "could",
    "will",
    "must",
}

_WORD = re.compile(r"[A-Za-z0-9']+")


def _norm(tok: str) -> str:
    """Normalize a token for case/punctuation-insensitive comparison."""
    return "".join(_WORD.findall(tok)).lower()


def _numbers(tokens: list[str]) -> list[str]:
    """Return numeric tokens exactly as they appear, in order."""
    return [
        w
        for t in tokens
        for w in _WORD.findall(t)
        if any(c.isdigit() for c in w)
    ]


def _negations(tokens: list[str]) -> set[str]:
    """Return negation words present in the token sequence."""
    return {
        w
        for t in tokens
        for w in _WORD.findall(t.lower())
        if w in _NEGATIONS
    }


def _hedges_commitments(tokens: list[str]) -> set[str]:
    """Return uncertainty/hedging/commitment words present in the text."""
    return {
        w
        for t in tokens
        for w in _WORD.findall(t.lower())
        if w in _HEDGES_COMMITMENTS
    }


def _match_model_change(
    o: str,
    r: str,
    changes,
) -> tuple[str, float] | None:
    """Match a diff span against an explicitly itemised model change.

    Matching is case/punctuation-insensitive, but the normalized original
    and refined spans must both match exactly. This prevents the model from
    using an audit entry for one edit to justify a different edit.
    """
    no, nr = _norm(o), _norm(r)

    for c in changes:
        if _norm(c.original) == no and _norm(c.refined) == nr:
            return c.reason, c.confidence

    return None


def _match_context(
    a: list[str],
    b: list[str],
    i1: int,
    i2: int,
    j1: int,
    j2: int,
    changes,
) -> tuple[str, float] | None:
    """Match a diff hunk against a model change that describes a LARGER span.

    The model may itemise "open I D Connect" -> "OpenID Connect" while the
    minimal diff hunk is only "open I D" -> "OpenID". This is accepted only if:

      model.original == left_ctx + hunk_original + right_ctx
      model.refined  == left_ctx + hunk_refined  + right_ctx

    where the left/right context are the SAME unchanged tokens in both the real
    original and the real refined text (case/punctuation-insensitive), and
    exactly one distinct model change fits. Otherwise the mapping is ambiguous
    and the hunk is treated as not itemised.
    """
    na = [_norm(t) for t in a]
    nb = [_norm(t) for t in b]
    fits: dict = {}

    for c in changes:
        mo = [_norm(t) for t in c.original.split()]
        mr = [_norm(t) for t in c.refined.split()]

        extra = len(mo) - (i2 - i1)
        if extra <= 0 or len(mr) - (j2 - j1) != extra:
            continue  # exact-span case is handled by _match_model_change

        for kl in range(0, min(i1, j1, extra) + 1):
            kr = extra - kl
            if i2 + kr > len(a) or j2 + kr > len(b):
                continue

            left_o, left_r = na[i1 - kl:i1], nb[j1 - kl:j1]
            right_o, right_r = na[i2:i2 + kr], nb[j2:j2 + kr]

            # Context must be unchanged on both sides of the hunk.
            if left_o != left_r or right_o != right_r:
                continue

            if (
                mo == left_o + na[i1:i2] + right_o
                and mr == left_r + nb[j1:j2] + right_r
            ):
                fits[(tuple(mo), tuple(mr), c.confidence)] = (
                    c.reason,
                    c.confidence,
                )
                break

    return next(iter(fits.values())) if len(fits) == 1 else None


def reconcile(
    original: str,
    seg: ModelSegment | None,
    min_conf: float,
):
    """Validate and reconcile one model-refined transcript segment.

    Returns:
        (
            refined_text,
            accepted_changes,
            rejected_count,
            warnings,
        )

    The original text is preserved whenever the model output is missing,
    empty, unsafe, or contains only rejected changes.
    """

    # No model output / empty output / unchanged output.
    if (
        seg is None
        or seg.refined_text.strip() == original.strip()
        or not seg.refined_text.strip()
    ):
        return (
            original,
            [],
            0,
            []
            if seg is not None
            else ["no model output for segment; kept original"],
        )

    # Word-level diff.
    a = original.split()
    b = seg.refined_text.split()

    sm = difflib.SequenceMatcher(
        a=a,
        b=b,
        autojunk=False,
    )

    out: list[str] = []
    changes: list[Change] = []
    rejected = 0
    warns: list[str] = []

    for op, i1, i2, j1, j2 in sm.get_opcodes():

        # No modification.
        if op == "equal":
            out.extend(a[i1:i2])
            continue

        o = a[i1:i2]
        r = b[j1:j2]

        ostr = " ".join(o)
        rstr = " ".join(r)

        # Check whether the model explicitly documented this change (exact span,
        # or a larger span whose surrounding context is unchanged).
        meta = _match_model_change(
            ostr,
            rstr,
            seg.changes,
        ) or _match_context(
            a,
            b,
            i1,
            i2,
            j1,
            j2,
            seg.changes,
        )

        if meta:
            reason, conf = meta
        else:
            reason = "modification not itemised by model"
            conf = 0.0

        # Pure capitalization/punctuation changes do not need a separate
        # model confidence entry.
        cosmetic = (
            [_norm(x) for x in o if _norm(x)]
            == [_norm(x) for x in r if _norm(x)]
        )

        if cosmetic and not meta:
            reason = "capitalisation/punctuation"
            conf = max(conf, min_conf)

        problem = None

        # 1. Numeric information must never change.
        if _numbers(o) != _numbers(r):
            problem = "changes numeric information"

        # 2. Negation must never change.
        elif _negations(o) != _negations(r):
            problem = "changes negation"

        # 3. Uncertainty / hedging / commitment language must never change.
        elif _hedges_commitments(o) != _hedges_commitments(r):
            problem = "changes uncertainty or commitment language"

        # 4. Non-cosmetic changes need sufficient confidence.
        elif conf < min_conf:
            problem = (
                f"confidence {conf:.2f} below threshold "
                f"{min_conf:.2f}"
            )

        if problem:
            rejected += 1

            warns.append(
                f"reverted '{ostr}' -> '{rstr}': {problem}"
            )

            # Preserve exactly what the original transcript contained.
            out.extend(o)

        else:
            # Accept the validated replacement.
            out.extend(r)

            changes.append(
                Change(
                    original=ostr,
                    refined=rstr,
                    reason=reason,
                    confidence=round(conf, 3),
                )
            )

    # If every proposed change was rejected, return the original transcript
    # rather than returning a partially reconstructed version.
    if not changes:
        return original, [], rejected, warns

    return " ".join(out), changes, rejected, warns