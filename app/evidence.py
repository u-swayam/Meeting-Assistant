"""Evidence layer: resolves LLM2 segment_ids against LLM1's refined_output.json.

Pure lookup over existing pipeline JSON. No model logic, nothing invented.
LLM2 segment_ids are positions in refined_output.json['segments'] (see llm2/documenter.py).
"""
from __future__ import annotations
import re

_WARN = re.compile(r"^(minutes|decisions|action_items)\[(\d+)\]:\s*(.*)$", re.S)


def _seg(segs, i):
    s = segs[i]
    d = {"id": i, "speaker": s["speaker"], "start": s["start"], "end": s["end"],
         "original_text": s["original_text"], "refined_text": s["refined_text"],
         "changed": s["changed"], "changes": s.get("changes", [])}
    if s.get("source_language"):          # non-English meeting: original_text IS the source-language segment `id`
        d["source_language"] = s["source_language"]
        d["translated"] = True
    return d


def build_evidence(refined: dict, doc: dict, transformation: dict | None = None) -> dict:
    segs = refined["segments"]
    n = len(segs)
    per_item: dict[tuple[str, int], list[str]] = {}
    general: list[str] = []
    for w in doc.get("warnings", []):
        m = _WARN.match(w)
        if m:
            per_item.setdefault((m.group(1), int(m.group(2))), []).append(m.group(3))
        else:
            general.append(w)

    items = []
    cands = (transformation or {}).get("decision_candidates") or [] if isinstance(transformation, dict) else []

    def consensus_for(ids):
        """Best-overlap decision candidate -> consensus payload with resolved segments. Pure lookup; None if absent."""
        best = max((c for c in cands if isinstance(c, dict) and c.get("consensus") and set(c.get("segment_ids", [])) & set(ids)),
                   key=lambda c: len(set(c["segment_ids"]) & set(ids)), default=None)
        if not best:
            return None
        ok = lambda L: [i for i in (L or []) if isinstance(i, int) and 0 <= i < n]
        cs = best["consensus"]
        groups = {k: [_seg(segs, i) for i in ok((cs.get("evidence") or {}).get(k))]
                  for k in ("decision_statement", "supporting_statements", "explicit_agreement",
                            "explicit_finalization", "opposition", "unresolved")}
        return {"decision_id": best.get("decision_id"), "candidate_status": best.get("status"),
                "classification": cs.get("classification"), "level": cs.get("level"), "score": cs.get("score"),
                "score_kind": cs.get("score_kind"), "notes": cs.get("notes", []), "evidence": groups,
                "reconstructed_text": best.get("reconstructed_text"), "reconstruction_source": best.get("reconstruction_source"),
                "original_text": best.get("original_text"), "trigger_segment": best.get("trigger_segment"),
                "context_segments": [_seg(segs, i) for i in ok(best.get("context_segments"))]}

    def add(kind, k, d, title, body):
        ids = [i for i in d.get("segment_ids", []) if isinstance(i, int) and 0 <= i < n]
        cited = [_seg(segs, i) for i in ids]
        speakers = list(dict.fromkeys(c["speaker"] for c in cited))
        items.append({
            "key": f"{kind}:{k}", "kind": kind, "index": k, "title": title, "body": body,
            "owner": d.get("owner") if kind == "action_items" else None,
            "deadline": d.get("deadline") if kind == "action_items" else None,
            "segment_ids": ids, "segments": cited, "speakers": speakers,
            "time_start": min((c["start"] for c in cited), default=None),
            "time_end": max((c["end"] for c in cited), default=None),
            "warnings": per_item.get((kind, k), []),
            # LLM2's schema has no status field, so none is claimed here.
            "status": None,
            "consensus": consensus_for(ids) if kind == "decisions" else None,
            "refinements_in_evidence": sum(len(c["changes"]) for c in cited),
        })

    for k, m in enumerate(doc.get("minutes", [])):
        add("minutes", k, m, m["topic"], m["discussion"])
    for k, d in enumerate(doc.get("decisions", [])):
        add("decisions", k, d, d["decision"], None)
    for k, a in enumerate(doc.get("action_items", [])):
        add("action_items", k, a, a["task"], None)
    return {"items": items, "general_warnings": general,
            "missing_fields": ["status (confirmed/proposed/pending/conditional) is not a field in "
                               "documentation_output.json; only validation warnings are available."]}
