from __future__ import annotations
from pathlib import Path

from llm2.schemas import DocumentationResult

DOC_JSON = "documentation_output.json"
DOC_TXT = "documentation_output.txt"


def to_text(res: DocumentationResult) -> str:
    """Human-readable view rendered FROM the validated JSON result (never a second LLM output)."""
    out = ["MEETING SUMMARY", "", res.summary, "", "MINUTES"]
    for i, m in enumerate(res.minutes, 1):
        out += ["", f"{i}. {m.topic}", f"   {m.discussion}"]
    out += ["", "KEY DECISIONS"]
    out += [f"- {d.decision}" for d in res.decisions] or ["- None confirmed in the recording."]
    out += ["", "ACTION ITEMS"]
    if not res.action_items:
        out.append("- None established in the recording.")
    for a in res.action_items:
        out.append(f"- {a.task}")
        out.append(f"    Owner: {a.owner or 'not stated'}   |   Deadline: {a.deadline or 'not stated'}")
    return "\n".join(out) + "\n"


def write_outputs(res: DocumentationResult, outdir: Path) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    jp, tp = outdir / DOC_JSON, outdir / DOC_TXT
    jp.write_text(res.model_dump_json(indent=2), encoding="utf-8")
    tp.write_text(to_text(res), encoding="utf-8")
    return jp, tp
