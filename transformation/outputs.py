from __future__ import annotations
from pathlib import Path

from transformation.schemas import TransformationResult

TRANSFORMATION_JSON = "transformation_output.json"


def write_outputs(res: TransformationResult, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / TRANSFORMATION_JSON
    p.write_text(res.model_dump_json(indent=2), encoding="utf-8")
    return p
