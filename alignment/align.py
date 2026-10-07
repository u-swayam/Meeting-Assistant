from __future__ import annotations
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, ConfigDict
from stt.schemas import STTResult
from diarization.schemas import DiarizationResult, DiarSegment


# ---------- output schema ----------
class Turn(BaseModel):
    model_config = ConfigDict(frozen=True)
    start: float
    end: float
    speaker: str
    text: str
    confidence: float = 1.0          # fraction of the turn covered by its speaker's diarization
    has_overlap: bool = False        # another speaker was also talking
    flags: tuple[str, ...] = ()      # overlap | split_segment | nearest_speaker | unknown_speaker | smoothed
    source_segment_ids: tuple[int, ...] = ()


def fmt_ts(t: float) -> str:
    tenths = int(round(max(t, 0.0) * 10))
    h, rem = divmod(tenths, 36000)
    m, rem = divmod(rem, 600)
    return f"{h:02d}:{m:02d}:{rem // 10:02d}.{rem % 10}"


class SpeakerLabelledTranscript(BaseModel):
    model_config = ConfigDict(frozen=True)
    turns: tuple[Turn, ...] = ()
    speakers: tuple[str, ...] = ()
    duration: float = 0.0
    warnings: tuple[str, ...] = ()

    def to_text(self) -> str:
        return "\n\n".join(f"[{fmt_ts(t.start)}] {t.speaker}\n{t.text}" for t in self.turns)


# ---------- config ----------
@dataclass(frozen=True)
class AlignConfig:
    max_gap: float = 0.6        # nearest-speaker fallback for words inside diarization gaps (s)
    min_run: float = 0.6        # runs shorter than this may be absorbed if weakly supported
    min_run_words: int = 3
    keep_conf: float = 0.5      # runs with confidence >= this are kept (real backchannels)
    overlap_frac: float = 0.3   # a speaker covering >= this fraction of a unit counts as "also speaking"
    merge_gap: float = 1.0      # merge same-speaker turns closer than this
    word_tol: float = 0.05
    unknown: str = "UNKNOWN"


# ---------- internals ----------
class _Timeline:
    def __init__(self, segs):
        self.segs: list[DiarSegment] = sorted(segs, key=lambda s: (s.start, s.end))
        self.starts = [s.start for s in self.segs]
        self.max_len = max((s.end - s.start for s in self.segs), default=0.0)

    def _cands(self, a: float, b: float):
        lo = bisect_left(self.starts, a - self.max_len)
        hi = bisect_right(self.starts, b)
        return self.segs[lo:hi]

    def overlaps(self, a: float, b: float) -> dict[str, float]:
        acc: dict[str, float] = {}
        for s in self._cands(a, b):
            o = min(b, s.end) - max(a, s.start)
            if o > 0:
                acc[s.speaker] = acc.get(s.speaker, 0.0) + o
        return acc

    def nearest(self, a: float, b: float, max_gap: float) -> Optional[str]:
        best = None
        for s in self._cands(a - max_gap, b + max_gap):
            d = max(s.start - b, a - s.end, 0.0)
            if d <= max_gap and (best is None or d < best[0]):
                best = (d, s.speaker)
        return best[1] if best else None


@dataclass
class _Unit:
    start: float
    end: float
    text: str
    seg_id: int
    is_word: bool
    speaker: Optional[str] = None
    conf: float = 0.0
    overlap: bool = False
    gap_fill: bool = False
    smoothed: bool = False

    @property
    def dur(self) -> float:
        return max(self.end - self.start, 1e-3)


def _assign(units: list[_Unit], primary: _Timeline, full: _Timeline, cfg: AlignConfig) -> None:
    prev: Optional[str] = None
    for u in units:
        qa, qb = u.start, max(u.end, u.start + 1e-3)
        ov = primary.overlaps(qa, qb)
        if ov:                                    # max-overlap vote, ties -> previous speaker
            best = max(ov.values())
            tied = sorted(s for s, v in ov.items() if v >= best - 1e-6)
            u.speaker = prev if prev in tied else tied[0]
            u.conf = min(1.0, ov[u.speaker] / u.dur)
        else:                                     # diarization gap -> nearest speaker or unknown
            u.speaker = primary.nearest(qa, qb, cfg.max_gap)
            u.conf = 0.0
            u.gap_fill = u.speaker is not None
        if u.speaker is not None:
            prev = u.speaker
        fo = full.overlaps(qa, qb)
        u.overlap = sum(1 for v in fo.values() if v >= cfg.overlap_frac * u.dur) >= 2


def _group(units: list[_Unit]) -> list[list[_Unit]]:
    runs: list[list[_Unit]] = []
    for u in units:
        if runs and runs[-1][-1].speaker == u.speaker:
            runs[-1].append(u)
        else:
            runs.append([u])
    return runs


def _run_conf(run: list[_Unit]) -> float:
    tot = sum(u.dur for u in run)
    return sum(u.conf * u.dur for u in run) / tot


def _smooth(units: list[_Unit], cfg: AlignConfig) -> None:
    """Absorb short, weakly supported runs (boundary jitter, tiny unknown islands) into a neighbour.
    Each pass strictly reduces the number of runs, so it terminates."""
    for _ in range(100000):
        runs = _group(units)
        idx = None
        for i, r in enumerate(runs):
            dur = r[-1].end - r[0].start
            if (len(runs) > 1 and dur < cfg.min_run and len(r) <= cfg.min_run_words
                    and _run_conf(r) < cfg.keep_conf):
                idx = i
                break
        if idx is None:
            return
        r = runs[idx]
        prev = runs[idx - 1] if idx > 0 else None
        nxt = runs[idx + 1] if idx + 1 < len(runs) else None
        if prev and nxt:
            if prev[0].speaker == nxt[0].speaker:
                spk = prev[0].speaker
            else:
                gp, gn = r[0].start - prev[-1].end, nxt[0].start - r[-1].end
                spk = prev[0].speaker if gp <= gn else nxt[0].speaker
        else:
            spk = (prev or nxt)[0].speaker
        for u in r:
            u.speaker, u.smoothed = spk, True


def _build_turns(units: list[_Unit], segs, cfg: AlignConfig) -> list[Turn]:
    by_seg: dict[int, list[_Unit]] = {}
    for u in units:
        by_seg.setdefault(u.seg_id, []).append(u)
    turns: list[Turn] = []
    for seg in segs:
        us = by_seg.get(seg.id, [])
        if not us:
            continue
        runs = _group(us)
        for r in runs:
            single = len(runs) == 1
            flags = set()
            if len(runs) > 1: flags.add("split_segment")
            if any(u.overlap for u in r): flags.add("overlap")
            if r[0].speaker is None: flags.add("unknown_speaker")
            if any(u.gap_fill and not u.smoothed for u in r): flags.add("nearest_speaker")
            if any(u.smoothed for u in r): flags.add("smoothed")
            turns.append(Turn(
                start=seg.start if single else r[0].start,
                end=seg.end if single else r[-1].end,
                speaker=r[0].speaker or cfg.unknown,
                text=seg.text if single else " ".join(u.text for u in r if u.text).strip(),
                confidence=round(_run_conf(r), 3),
                has_overlap="overlap" in flags,
                flags=tuple(sorted(flags)),
                source_segment_ids=(seg.id,)))
    return turns


def _merge(turns: list[Turn], cfg: AlignConfig) -> list[Turn]:
    out: list[Turn] = []
    for t in turns:
        if out and out[-1].speaker == t.speaker and t.start - out[-1].end <= cfg.merge_gap:
            p = out[-1]
            wp, wt = max(p.end - p.start, 1e-3), max(t.end - t.start, 1e-3)
            out[-1] = Turn(
                start=p.start, end=max(p.end, t.end), speaker=p.speaker,
                text=f"{p.text} {t.text}".strip(),
                confidence=round((p.confidence * wp + t.confidence * wt) / (wp + wt), 3),
                has_overlap=p.has_overlap or t.has_overlap,
                flags=tuple(sorted(set(p.flags) | set(t.flags))),
                source_segment_ids=p.source_segment_ids + t.source_segment_ids)
        else:
            out.append(t)
    return out


# ---------- public API ----------
def align_transcript(stt: STTResult, diar: DiarizationResult,
                     config: Optional[AlignConfig] = None) -> SpeakerLabelledTranscript:
    cfg = config or AlignConfig()
    warnings = list(stt.warnings) + list(diar.warnings)
    segs = sorted(stt.segments, key=lambda s: (s.start, s.id))
    if not segs:
        return SpeakerLabelledTranscript(duration=stt.duration,
                                         warnings=tuple(warnings + ["STT produced no segments."]))
    if not diar.segments:
        warnings.append("Diarization produced no segments; all turns are UNKNOWN.")

    full = _Timeline(diar.segments)
    primary = _Timeline(diar.exclusive_segments) if diar.exclusive_segments else full

    words = sorted(stt.words, key=lambda w: (w.start + w.end) / 2)
    mids = [(w.start + w.end) / 2 for w in words]
    taken: set[int] = set()
    units: list[_Unit] = []
    for seg in segs:
        lo = bisect_left(mids, seg.start - cfg.word_tol)
        hi = bisect_right(mids, seg.end + cfg.word_tol)
        idx = [i for i in range(lo, hi) if i not in taken]
        taken.update(idx)
        if idx:
            units += [_Unit(words[i].start, words[i].end, words[i].text, seg.id, True) for i in idx]
        else:
            units.append(_Unit(seg.start, seg.end, seg.text, seg.id, False))

    _assign(units, primary, full, cfg)
    _smooth(units, cfg)
    turns = _merge(_build_turns(units, segs, cfg), cfg)
    speakers = tuple(sorted({t.speaker for t in turns if t.speaker != cfg.unknown}))
    return SpeakerLabelledTranscript(turns=tuple(turns), speakers=speakers,
                                     duration=stt.duration, warnings=tuple(warnings))
