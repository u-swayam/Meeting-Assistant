from __future__ import annotations
import json
from pathlib import Path

from llm1.config import LLM1Config          # shared provider/env configuration
from llm1.providers import build_client     # shared DeepSeek client - no second client
from transformation.errors import TransformationError
from transformation.outputs import write_outputs
from transformation.prompts import SYSTEM_PROMPT, build_user_prompt
from transformation.schemas import TransformationResult
from transformation.validation import validate_response


def transform_transcript(data: dict, cfg: LLM1Config | None = None, client=None,
                         source_file: str = "") -> TransformationResult:
    """Structured representation of a refined transcript dict. Segment ids = positions in data['segments']."""
    cfg = cfg or LLM1Config.from_env()
    client = client or build_client(cfg)
    segs = data.get("segments") or []
    if not segs:
        raise TransformationError("refined transcript has no segments")
    segments = [{"speaker": s["speaker"], "start": s["start"], "end": s["end"], "text": s["refined_text"]}
                for s in segs]
    prompt = build_user_prompt(segments)
    warnings: list[str] = []
    parsed = None
    for attempt in (1, 2):                                   # one retry, same as LLM2
        raw = client.generate_json(SYSTEM_PROMPT, prompt)
        try:
            parsed, w = validate_response(raw, segments)
            warnings += w
            break
        except TransformationError as e:
            warnings.append(f"attempt {attempt}: {e}")
    if parsed is None:
        raise TransformationError("transformation output unusable after 2 attempts: " + "; ".join(warnings)
                                  + " (if truncated, raise DEEPSEEK_MAX_TOKENS)")
    warnings += list(getattr(client, "events", []))
    return TransformationResult(provider=cfg.provider, model=cfg.model, source_file=source_file,
                                total_segments=len(segments), warnings=warnings, **parsed)


def transform_file(refined_json: str | Path, outdir: str | Path | None = None,
                   cfg: LLM1Config | None = None, client=None):
    """Read refined_output.json (never modified), write transformation_output.json. Nothing is written on failure."""
    p = Path(refined_json)
    data = json.loads(p.read_text(encoding="utf-8"))
    res = transform_transcript(data, cfg, client, source_file=p.name)
    return res, write_outputs(res, Path(outdir) if outdir else p.parent)
