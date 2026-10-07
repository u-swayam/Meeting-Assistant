"""Count-agnostic clean-up of a diarization (pure Python, no torch/pydantic).

Works on plain (start, end, speaker) tuples so it is easy to test. It never needs the number of
speakers: it only (1) merges speakers whose voice centroids are near-identical and (2) absorbs
"micro-speakers" (a spurious extra cluster with only a few seconds of speech). A tiny speaker whose
voice is clearly distinct from everyone else is KEPT (it may be a real short-talking participant).
All thresholds are heuristics: calibrate them with tests/diag_diarization.py on your own recordings.
"""
from __future__ import annotations
import math, os
from dataclasses import dataclass
from typing import Optional

Seg = tuple[float, float, str]  # start, end, speaker


def _env_f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "")) if os.environ.get(name, "").strip() else default
    except ValueError:
        return default


@dataclass(frozen=True)
class PostConfig:
    spurious_abs_s: float = 4.0         # a speaker with less total speech than min(this, frac*total) is "tiny"
    spurious_frac: float = 0.04
    spurious_abs_no_emb_s: float = 2.0  # stricter limit when no voice centroids are available
    absorb_min_sim: float = 0.30        # tiny speaker is absorbed only if its nearest centroid has cosine >= this
    merge_sim: float = 0.80             # merge two speakers whose centroids have cosine >= this
    merge_gap: float = 0.05             # re-join same-speaker segments separated by <= this (s)
    relabel: bool = True                # close label gaps (00, 02 -> 00, 01)

    @classmethod
    def from_env(cls) -> "PostConfig":
        d = cls()
        return cls(spurious_abs_s=_env_f("DIARIZATION_SPURIOUS_ABS_S", d.spurious_abs_s),
                   spurious_frac=_env_f("DIARIZATION_SPURIOUS_FRAC", d.spurious_frac),
                   spurious_abs_no_emb_s=_env_f("DIARIZATION_SPURIOUS_ABS_NO_EMB_S", d.spurious_abs_no_emb_s),
                   absorb_min_sim=_env_f("DIARIZATION_ABSORB_MIN_SIM", d.absorb_min_sim),
                   merge_sim=_env_f("DIARIZATION_MERGE_SIM", d.merge_sim))


@dataclass
class PostResult:
    full: list[Seg]
    exclusive: list[Seg]
    mapping: dict[str, str]
    notes: list[str]


def cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def durations(segs: list[Seg]) -> dict[str, float]:
    d: dict[str, float] = {}
    for s, e, k in segs:
        d[k] = d.get(k, 0.0) + max(0.0, e - s)
    return d


def _redirect(mapping: dict[str, str], drop: str, keep: str) -> None:
    for k, v in mapping.items():
        if v == drop:
            mapping[k] = keep


def _neighbour_vote(segs: list[Seg], speaker: str) -> Optional[str]:
    """Speaker adjacent in time to `speaker`'s segments (duration-weighted vote)."""
    segs = sorted(segs, key=lambda x: (x[0], x[1]))
    votes: dict[str, float] = {}
    for i, (s, e, k) in enumerate(segs):
        if k != speaker:
            continue
        prev = next(((s - pe, pk) for ps, pe, pk in reversed(segs[:i]) if pk != speaker), None)
        nxt = next(((ns - e, nk) for ns, ne, nk in segs[i + 1:] if nk != speaker), None)
        cands = [c for c in (prev, nxt) if c is not None]
        if cands:
            votes[min(cands)[1]] = votes.get(min(cands)[1], 0.0) + max(e - s, 1e-3)
    return max(votes, key=votes.get) if votes else None


def _merge_touching(segs: list[Seg], gap: float) -> list[Seg]:
    by: dict[str, list[list[float]]] = {}
    for s, e, k in sorted(segs, key=lambda x: (x[0], x[1])):
        runs = by.setdefault(k, [])
        if runs and s - runs[-1][1] <= gap:
            runs[-1][1] = max(runs[-1][1], e)
        else:
            runs.append([s, e])
    out = [(a, b, k) for k, runs in by.items() for a, b in runs]
    return sorted(out, key=lambda x: (x[0], x[1], x[2]))


def postprocess(full: list[Seg], exclusive: list[Seg],
                centroids: Optional[dict[str, list[float]]] = None,
                cfg: Optional[PostConfig] = None) -> PostResult:
    cfg = cfg or PostConfig()
    notes: list[str] = []
    basis = exclusive or full                      # exclusive has no double-counted overlap
    dur = durations(basis)
    for _, _, k in full:                           # speakers only present in the overlapping view
        dur.setdefault(k, 0.0)
    original = sorted(dur)
    mapping = {k: k for k in dur}
    cent = {k: list(v) for k, v in (centroids or {}).items() if k in dur}

    # 1) merge speakers with near-identical voices
    while len(cent) >= 2:
        keys = sorted(cent)
        best = max(((cosine(cent[a], cent[b]), a, b) for i, a in enumerate(keys) for b in keys[i + 1:]))
        if best[0] < cfg.merge_sim:
            break
        sim, a, b = best
        keep, drop = (a, b) if dur[a] >= dur[b] else (b, a)
        w1, w2 = max(dur[keep], 1e-6), max(dur[drop], 1e-6)
        cent[keep] = [(x * w1 + y * w2) / (w1 + w2) for x, y in zip(cent[keep], cent[drop])]
        dur[keep] += dur[drop]
        del dur[drop], cent[drop]
        _redirect(mapping, drop, keep)
        notes.append(f"merged {drop} into {keep} (centroid cosine {sim:.2f})")

    # 2) absorb micro-speakers
    total = sum(dur.values())
    kept_distinct: set[str] = set()
    while len(dur) > 1:
        limit = min(cfg.spurious_abs_s if cent else cfg.spurious_abs_no_emb_s, cfg.spurious_frac * total)
        cands = sorted((d, k) for k, d in dur.items() if d < limit and k not in kept_distinct)
        if not cands:
            break
        d, k = cands[0]
        target, why = None, ""
        if cent and k in cent and len(cent) > 1:
            sim, o = max((cosine(cent[k], cent[o]), o) for o in cent if o != k)
            if sim >= cfg.absorb_min_sim:
                target, why = o, f"nearest centroid cosine {sim:.2f}"
            else:
                kept_distinct.add(k)
                notes.append(f"kept tiny {k} ({d:.1f}s): voice distinct (best cosine {sim:.2f})")
                continue
        elif not cent:
            cur = [(s, e, mapping[x]) for s, e, x in basis if mapping.get(x, x) in dur]
            target, why = _neighbour_vote(cur, k), "temporal neighbour"
        if target is None:
            kept_distinct.add(k)
            continue
        dur[target] += dur.pop(k)
        cent.pop(k, None)
        _redirect(mapping, k, target)
        notes.append(f"absorbed {k} ({d:.1f}s) into {target} ({why})")

    # 3) relabel to close gaps, apply mapping
    final = sorted(set(mapping.values()))
    if cfg.relabel and len(final) < len(original):
        rename = {old: f"SPEAKER_{i:02d}" for i, old in enumerate(final)}
        mapping = {k: rename[v] for k, v in mapping.items()}
        notes.append("relabelled speakers: " + ", ".join(f"{o}->{n}" for o, n in rename.items() if o != n))

    def apply(segs: list[Seg]) -> list[Seg]:
        return _merge_touching([(s, e, mapping.get(k, k)) for s, e, k in segs], cfg.merge_gap)

    return PostResult(apply(full), apply(exclusive), mapping, notes)
