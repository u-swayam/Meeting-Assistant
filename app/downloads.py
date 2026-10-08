"""Download artifacts for a meeting (stdlib only).

Every artifact is either an EXISTING file in the meeting's output directory or a deterministic text view rendered
from the existing JSON (summary / minutes / decisions / action items). No LLM is ever called here, and the text
matches what the UI displays because both read the same documentation_output.json.
"""
from __future__ import annotations
import json
from pathlib import Path

TXT, JSON = "text/plain; charset=utf-8", "application/json; charset=utf-8"

# key -> (file name used on disk / for the download, content type, label)
ARTIFACTS: dict[str, tuple[str, str, str]] = {
    "raw_transcript":     ("raw_transcript.txt", TXT, "Raw transcript"),
    "refined_transcript": ("refined_transcript.txt", TXT, "Refined transcript"),
    "minutes":            ("minutes.txt", TXT, "Meeting minutes"),
    "summary":            ("summary.txt", TXT, "Summary"),
    "decisions":          ("decisions.txt", TXT, "Decisions"),
    "action_items":       ("action_items.txt", TXT, "Action items"),
    "documentation_json": ("documentation_output.json", JSON, "Final record (JSON)"),
    "refined_json":       ("refined_output.json", JSON, "Refined transcript (JSON)"),
    "transformation_json": ("transformation_output.json", JSON, "Transformation (JSON)"),
    "aligned_json":       ("final_output.json", JSON, "Aligned transcript (JSON)"),
}
DOC_DERIVED = {"minutes", "summary", "decisions", "action_items"}
STANDALONE_FILES = {k: ARTIFACTS[k][0] for k in DOC_DERIVED}


class ArtifactError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message); self.status, self.message = status, message


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    except (OSError, ValueError):
        return None


# ---- deterministic renderers over documentation_output.json ----
def render_summary(doc: dict) -> str:
    return (doc.get("summary") or "").strip() + "\n"


def render_minutes(doc: dict) -> str:
    out = ["MINUTES"]
    for i, m in enumerate(doc.get("minutes", []), 1):
        out += ["", f"{i}. {m.get('topic', '')}", f"   {m.get('discussion', '')}"]
    if not doc.get("minutes"):
        out += ["", "No minutes were generated."]
    return "\n".join(out) + "\n"


def render_decisions(doc: dict) -> str:
    # Only what LLM2 listed as final decisions; proposals/unresolved items live in the minutes, never here.
    out = ["KEY DECISIONS", ""]
    ds = doc.get("decisions", [])
    out += [f"{i}. {d.get('decision', '')}" for i, d in enumerate(ds, 1)] or ["None confirmed in the recording."]
    return "\n".join(out) + "\n"


def render_action_items(doc: dict) -> str:
    out = ["ACTION ITEMS", ""]
    items = doc.get("action_items", [])
    if not items:
        out.append("None established in the recording.")
    for i, a in enumerate(items, 1):
        out += [f"{i}. {a.get('task', '')}",
                f"   Owner: {a.get('owner') or 'not stated'}",
                f"   Deadline: {a.get('deadline') or 'not stated'}"]
        if a.get("condition"):                        # shown whenever the record carries one
            out.append(f"   Condition: {a['condition']}")
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


_RENDER = {"summary": render_summary, "minutes": render_minutes,
           "decisions": render_decisions, "action_items": render_action_items}


def write_standalone(doc: dict, outdir: Path) -> list[Path]:
    """Write summary/minutes/decisions/action_items .txt next to documentation_output.json (no LLM)."""
    paths = []
    for key, fn in _RENDER.items():
        p = outdir / STANDALONE_FILES[key]
        p.write_text(fn(doc), encoding="utf-8"); paths.append(p)
    return paths


# ---- lookup ----
def _raw_text(d: Path) -> str | None:
    p = d / "raw_transcript.txt"
    if p.is_file():
        return p.read_text(encoding="utf-8")
    fo = _load(d / "final_output.json")                  # same raw STT text, stored by the pipeline
    t = fo.get("raw_transcript") if isinstance(fo, dict) else None
    return t if isinstance(t, str) and t else None


def _refined_text(d: Path) -> str | None:
    p = d / "refined_transcript.txt"
    if p.is_file():
        return p.read_text(encoding="utf-8")
    r = _load(d / "refined_output.json")                 # same join llm1.outputs.to_text uses
    return "\n\n".join(s["refined_text"] for s in r["segments"]) if isinstance(r, dict) and r.get("segments") else None


def read_artifact(d: Path, key: str) -> tuple[bytes, str, str]:
    """-> (body, content_type, file_name). Raises ArtifactError(400/404) with a clear message."""
    if key not in ARTIFACTS:
        raise ArtifactError(400, f"Unknown artifact {key!r}. Allowed: {', '.join(sorted(ARTIFACTS))}.")
    fname, ctype, label = ARTIFACTS[key]
    text: str | None = None
    if key == "raw_transcript":
        text = _raw_text(d)
    elif key == "refined_transcript":
        text = _refined_text(d)
    elif key in DOC_DERIVED:
        doc = _load(d / "documentation_output.json")
        text = _RENDER[key](doc) if isinstance(doc, dict) else None
    else:
        p = d / fname
        if p.is_file():
            return p.read_bytes(), ctype, fname
    if text is None:
        raise ArtifactError(404, f"{label} is not available for this meeting yet "
                                 f"(the stage that produces it has not completed).")
    return text.encode("utf-8"), ctype, fname


def availability(d: Path) -> dict[str, bool]:
    out = {}
    for k in ARTIFACTS:
        try:
            read_artifact(d, k); out[k] = True
        except ArtifactError:
            out[k] = False
    return out
