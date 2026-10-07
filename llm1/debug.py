"""Opt-in diagnostics (LLM1_DEBUG_RAW_RESPONSE=true). Writes what was SENT to and RECEIVED from the model,
per chunk, so a run can be diagnosed as "model returned no corrections" vs "our code discarded them".

Never writes API keys: configured secrets are scrubbed from every file, and the HTTP request/headers are
never captured (only the messages and non-secret request parameters).
Writing is best-effort: a failure here is recorded in `errors` and never changes the refinement result."""
from __future__ import annotations

import json
import time
from pathlib import Path

_TRUTHY = {"1", "true", "yes", "on"}


def flag_enabled(value) -> bool:
    return str(value or "").strip().lower() in _TRUTHY


class DebugRecorder:
    def __init__(self, debug_dir, secrets=()):
        self.dir = Path(debug_dir)
        self.secrets = [s for s in secrets if s and len(s) >= 6]
        self.run_id = time.strftime("%Y%m%dT%H%M%S")
        self.errors: list[str] = []

    def _scrub(self, text: str) -> str:
        for s in self.secrets:
            text = text.replace(s, "***")
        return text

    def _write(self, name: str, obj) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            text = self._scrub(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
            (self.dir / name).write_text(text, encoding="utf-8")
        except Exception as e:  # noqa: BLE001 - diagnostics must never break the run
            self.errors.append(f"debug write failed for {name}: {type(e).__name__}")

    @staticmethod
    def _head(label, chunk):
        no = label.split("_")[1] if label.startswith("chunk_") else label
        return {"chunk_number": int(no) if str(no).isdigit() else no, "label": label,
                "requested_indices": [c["index"] for c in chunk]}

    def record_request(self, label, chunk, system_prompt, user_prompt) -> None:
        self._write(f"{label}_request.json", {
            "run_id": self.run_id, **self._head(label, chunk),
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}],
            "requested_segments": chunk})

    def record_response(self, label, chunk, client, parsed, error=None) -> None:
        raw = getattr(client, "last_raw", None)
        doc = {"run_id": self.run_id, **self._head(label, chunk),
               "request_params": getattr(client, "last_request_params", None),
               "provider_exposes_raw_content": raw is not None,
               "raw_response": raw,                       # content / finish_reason / model / usage
               "parsed_json": parsed, "call_error": error,
               "analysis": self._analyse(chunk, parsed)}
        self._write(f"{label}_raw.json", doc)

    @staticmethod
    def _analyse(chunk, parsed):
        """How many segments did the MODEL itself change, before any of our validation?"""
        segs = parsed.get("segments") if isinstance(parsed, dict) else None
        if not isinstance(segs, list):
            return {"note": "no 'segments' list in parsed response"}
        orig = {c["index"]: c["text"] for c in chunk}
        rows = []
        for s in segs:
            if not isinstance(s, dict):
                continue
            i, txt = s.get("index"), s.get("refined_text")
            ch = s.get("changes")
            rows.append({"index": i,
                         "refined_text_differs_from_input": (isinstance(txt, str) and i in orig
                                                             and txt.strip() != orig[i].strip()),
                         "itemised_changes": len(ch) if isinstance(ch, list) else None})
        return {"segments_returned": len(rows),
                "segments_where_model_text_differs": sum(1 for r in rows if r["refined_text_differs_from_input"]),
                "total_itemised_changes": sum(r["itemised_changes"] or 0 for r in rows),
                "per_segment": rows}