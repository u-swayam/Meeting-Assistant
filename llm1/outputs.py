from __future__ import annotations

from pathlib import Path

from llm1.schemas import RefinementResult


REFINED_JSON = "refined_output.json"
REFINED_TXT = "refined_transcript.txt"


def _ts(t: float) -> str:
    """Format seconds as HH:MM:SS.t, matching alignment.align.fmt_ts."""
    tenths = int(round(max(t, 0.0) * 10))
    h, rem = divmod(tenths, 36000)
    m, rem = divmod(rem, 600)

    return f"{h:02d}:{m:02d}:{rem // 10:02d}.{rem % 10}"


def to_text(res: RefinementResult) -> str:
    """Create a clean human-readable refined transcript without timestamps
    or speaker labels. Metadata remains preserved in refined_output.json.
    """
    return "\n\n".join(
        s.refined_text
        for s in res.segments
    )


def write_outputs(
    res: RefinementResult,
    outdir: Path,
) -> tuple[Path, Path]:
    """Write refined JSON and human-readable refined TXT."""
    outdir.mkdir(parents=True, exist_ok=True)

    jp = outdir / REFINED_JSON
    tp = outdir / REFINED_TXT

    jp.write_text(
        res.model_dump_json(indent=2),
        encoding="utf-8",
    )

    tp.write_text(
        to_text(res),
        encoding="utf-8",
    )

    return jp, tp