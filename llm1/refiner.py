from __future__ import annotations
import json
from pathlib import Path

from pydantic import ValidationError

from llm1.config import LLM1Config, LLM1ProviderError
from llm1.debug import DebugRecorder
from llm1.providers import build_client
from llm1.outputs import write_outputs
from llm1.prompts import SYSTEM_PROMPT, build_user_prompt
from llm1.schemas import ModelResponse, RefinedSegment, RefinementResult, derive_meta
from llm1.validation import reconcile

_META_KEYS = ("confidence", "has_overlap", "flags", "source_segment_ids")


def refine_transcript(data: dict, cfg: LLM1Config | None = None, client=None,
                      source_file: str = "", debug_dir: str | Path | None = None) -> RefinementResult:
    """Refine the `turns` of a structured transcript dict. Input is never mutated."""
    cfg = cfg or LLM1Config.from_env()
    client = client or build_client(cfg)
    turns = data.get("turns") or []
    warnings: list[str] = []
    model_out = {}
    n = max(cfg.chunk_turns, 1)
    rec = None
    if debug_dir and cfg.debug_raw_response:      # diagnostics only; off => no behaviour change
        rec = DebugRecorder(debug_dir, secrets=[cfg.deepseek_api_key, cfg.api_key, cfg.groq_api_key,
                                                  cfg.cerebras_api_key, cfg.openrouter_api_key])

    def request(chunk, label="chunk_00"):
        """One API call. Returns {index: ModelSegment} only if indices match the chunk exactly."""
        user_prompt = build_user_prompt(chunk, len(turns))
        if rec:
            rec.record_request(label, chunk, SYSTEM_PROMPT, user_prompt)
        try:
            raw = client.generate_json(SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            if rec:
                rec.record_response(label, chunk, client, None, error=f"{type(e).__name__}: {e}")
            raise
        if rec:
            rec.record_response(label, chunk, client, raw)
        try:
            resp = ModelResponse.model_validate(raw)
        except ValidationError as e:
            return None, f"schema-invalid ({str(e)[:120]})"
        got = [m.index for m in resp.segments]
        want = [c["index"] for c in chunk]
        if sorted(got) != want:
            missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
            dup = len(got) - len(set(got))
            return None, f"indices mismatch (missing={missing[:8]}, extra={extra[:8]}, duplicates={dup})"
        return {m.index: m for m in resp.segments}, ""

    for s in range(0, len(turns), n):
        chunk = [{"index": i, "speaker": t["speaker"], "start": t["start"], "end": t["end"],
                  "text": t["text"]} for i, t in enumerate(turns[s:s + n], start=s)]
        label = f"chunk_{s // n:02d}"
        got, why = request(chunk, label)
        if got is not None:
            model_out.update(got)
            continue
        # one retry with a smaller chunk size (halves); a single segment is retried as-is
        mid = len(chunk) // 2
        parts = [chunk[:mid], chunk[mid:]] if mid else [chunk]
        warnings.append(f"segments {chunk[0]['index']}-{chunk[-1]['index']}: {why}; "
                        f"retrying with {len(parts)} smaller request(s)")
        for k, part in enumerate(parts):
            got, why = request(part, f"{label}_retry_{k}")
            if got is None:
                warnings.append(f"segments {part[0]['index']}-{part[-1]['index']}: {why}; "
                                f"kept original text unchanged")
            else:
                model_out.update(got)

    warnings += list(getattr(client, "events", []))
    if rec:
        warnings += rec.errors
    segs, rejected = [], 0
    for i, t in enumerate(turns):
        text, changes, rej, w = reconcile(t["text"], model_out.get(i), cfg.min_confidence)
        rejected += rej
        warnings += [f"segment {i}: {x}" for x in w]
        segs.append(RefinedSegment(
            start=t["start"], end=t["end"], speaker=t["speaker"],
            original_text=t["text"], refined_text=text,
            changed=bool(changes), changes=changes,
            metadata={k: t[k] for k in _META_KEYS if k in t}))
    return RefinementResult(
        provider=cfg.provider, model=cfg.model, source_file=source_file,
        total_segments=len(segs), changed_segments=sum(s.changed for s in segs),
        total_changes=sum(len(s.changes) for s in segs), rejected_changes=rejected,
        speakers=derive_meta(segs, data.get("speakers"), data.get("duration"))[0],
        duration=derive_meta(segs, data.get("speakers"), data.get("duration"))[1],
        warnings=warnings + list(data.get("warnings", [])), segments=segs)


def refine_file(structured_json: str | Path, outdir: str | Path | None = None,
                cfg: LLM1Config | None = None, client=None) -> tuple[RefinementResult, Path, Path]:
    """Read the structured transcript, refine, write refined JSON + TXT next to it."""
    p = Path(structured_json)
    data = json.loads(p.read_text(encoding="utf-8"))
    cfg = cfg or LLM1Config.from_env()
    out = Path(outdir) if outdir else p.parent
    res = refine_transcript(data, cfg, client, source_file=p.name, debug_dir=out / "debug")
    jp, tp = write_outputs(res, out)
    return res, jp, tp