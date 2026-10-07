from __future__ import annotations
import json
from pathlib import Path

from llm1.config import LLM1Config          # shared provider/env configuration
from llm1.providers import build_client     # shared DeepSeek client (same model as LLM1)
from llm2.errors import LLM2OutputError
from llm2.guidance import GUIDANCE_ADDENDUM, cross_check, render_guidance
from llm2.outputs import write_outputs
from llm2.prompts import SYSTEM_PROMPT, build_user_prompt
from llm2.schemas import DocumentationResult
from llm2.validation import validate_response


def document_transcript(data: dict, cfg: LLM1Config | None = None, client=None,
                        source_file: str = "", transformation: dict | None = None,
                        transformation_source: str | None = None) -> DocumentationResult:
    """Build documentation from a refined transcript dict (LLM1's refined_output.json).

    Segment ids in the output are positions in data['segments'].
    If `transformation` (transformation_output.json as a dict) is given it is added as NON-authoritative
    guidance; the refined transcript remains the source of truth. Without it behaviour is unchanged.
    """
    cfg = cfg or LLM1Config.from_env()
    client = client or build_client(cfg)
    segs = data.get("segments") or []
    if not segs:
        raise LLM2OutputError("refined transcript has no segments")
    segments = [{"speaker": s["speaker"], "start": s["start"], "end": s["end"],
                 "text": s["refined_text"]} for s in segs]
    user_prompt = build_user_prompt(segments)
    system_prompt = SYSTEM_PROMPT
    if transformation:
        user_prompt += "\n" + render_guidance(transformation)
        system_prompt = SYSTEM_PROMPT + GUIDANCE_ADDENDUM

    warnings: list[str] = []
    parsed = None
    for attempt in (1, 2):                         # one retry on unusable output, then fail clearly
        raw = client.generate_json(system_prompt, user_prompt)
        try:
            parsed, w = validate_response(raw, segments)
            warnings += w
            break
        except LLM2OutputError as e:
            warnings.append(f"attempt {attempt}: {e}")
    if parsed is None:
        raise LLM2OutputError("LLM2 output unusable after 2 attempts: " + "; ".join(warnings)
                              + " (if truncated, raise DEEPSEEK_MAX_TOKENS)")
    if transformation:
        warnings += cross_check(parsed, transformation)
    warnings += list(getattr(client, "events", []))
    return DocumentationResult(provider=cfg.provider, model=cfg.model, source_file=source_file,
                               total_segments=len(segments), warnings=warnings,
                               transformation_source=transformation_source if transformation else None, **parsed)


def document_file(refined_json: str | Path, outdir: str | Path | None = None,
                  cfg: LLM1Config | None = None, client=None, transformation_path: str | Path | None = None):
    """Read refined_output.json, write documentation_output.json then documentation_output.txt.

    If transformation_output.json exists next to the refined file (or `transformation_path` is given) it is
    used as guidance; otherwise LLM2 runs exactly as before (backward compatible)."""
    p = Path(refined_json)
    data = json.loads(p.read_text(encoding="utf-8"))
    tp_ = Path(transformation_path) if transformation_path else p.parent / "transformation_output.json"
    trans = json.loads(tp_.read_text(encoding="utf-8")) if tp_.is_file() else None
    res = document_transcript(data, cfg, client, source_file=p.name, transformation=trans,
                              transformation_source=tp_.name if trans else None)
    jp, tp = write_outputs(res, Path(outdir) if outdir else p.parent)
    return res, jp, tp
