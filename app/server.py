"""Meeting Assistant UI server (stdlib only; reads the pipeline's existing JSON outputs).

Run from the project root:   python -m app.server [--port 8000] [--outputs outputs]
"""
from __future__ import annotations
import argparse, json, re, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from app.evidence import build_evidence
from app.downloads import ArtifactError, availability, read_artifact
from common.errors import UnsupportedLanguageError
from app import pipeline_runner as pr
from app import ask as ask_mod
from stt.languages import supported_languages

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
OUTPUTS = ROOT / "outputs"
UPLOADS = ROOT / "uploads"
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".webm", ".aac"}


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def meeting_dir(mid: str) -> Path | None:
    d = (OUTPUTS / mid)
    return d if re.fullmatch(r"[\w.\- ]+", mid) and d.is_dir() and d.parent == OUTPUTS else None


def list_meetings():
    out = []
    for d in sorted(OUTPUTS.iterdir()) if OUTPUTS.is_dir() else []:
        if d.is_dir():
            out.append({
                "id": d.name, 
                "has_raw": (d / "raw_transcript.txt").is_file() or (d / "final_output.json").is_file(),
                "has_refined": (d / "refined_output.json").is_file(),
                "has_transformation": (d / "transformation_output.json").is_file(),
                "has_documentation": (d / "documentation_output.json").is_file()
            })
    return out


def bundle(mid: str):
    d = meeting_dir(mid)
    if not d:
        return None
    raw, refined, doc = (_load(d / "final_output.json"), _load(d / "refined_output.json"),
                         _load(d / "documentation_output.json"))
    trans = _load(d / "transformation_output.json")
    ev = build_evidence(refined, doc, trans) if refined and doc else None
    aligned = bool(raw and refined and len(raw.get("turns", [])) == len(refined["segments"]) and
                   all(t["text"] == s["original_text"] for t, s in zip(raw["turns"], refined["segments"])))
    
    # Grab the most recent job for this meeting to show live status/errors
    job = next((j for j in reversed(list(pr.JOBS.values())) if j["meeting_id"] == mid), None)
    
    lang = (raw or {}).get("language")           # present only for non-English meetings
    translation = ({"source_language": refined.get("source_language"), "source_language_name": refined.get("source_language_name"),
                    "target_language": refined.get("target_language", "en"), "stt_provider": refined.get("stt_provider"),
                    "refinement": f"{refined.get('provider')}/{refined.get('model')}"}
                   if refined and refined.get("source_language") else None)
    return {"id": mid, "language": lang, "translation": translation, "downloads": availability(d), "raw": raw, "refined": refined, "documentation": doc, "evidence": ev,
            "transformation": trans,
            "raw_text": (d / "raw_transcript.txt").read_text(encoding="utf-8") if (d / "raw_transcript.txt").is_file() else None,
            "raw_aligned_with_refined": aligned,
            "job": job}


class H(BaseHTTPRequestHandler):
    def _send(self, code, body: bytes, ctype="application/json", headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if headers:
            for k, v in headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def log_message(self, fmt, *a):
        sys.stderr.write("%s\n" % (fmt % a))

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/meetings":
            return self._json(list_meetings())
        if path == "/api/languages":
            return self._json(supported_languages())
        
        m = re.fullmatch(r"/api/meetings/([^/]+)", path)
        if m:
            b = bundle(m.group(1))
            return self._json(b) if b else self._json({"error": "meeting not found"}, 404)
        
        m = re.fullmatch(r"/api/jobs/([\w]+)", path)
        if m:
            j = pr.JOBS.get(m.group(1))
            return self._json(j) if j else self._json({"error": "job not found"}, 404)

        # DOWNLOAD ENDPOINTS
        m = re.fullmatch(r"/api/download/([^/]+)/([^/]+)", path)
        if m:
            mid, artifact = m.groups()
            d = meeting_dir(mid)
            if not d:
                return self._json({"error": "meeting not found"}, 404)

            # legacy UI keys -> downloads.ARTIFACTS keys
            key = {"raw": "raw_transcript", "refined": "refined_transcript", "actions": "action_items",
                   "json": "documentation_json", "transformation": "transformation_json"}.get(artifact, artifact)
            try:
                body, ctype, fname = read_artifact(d, key)
            except ArtifactError as e:
                return self._json({"error": e.message}, e.status)
            return self._send(200, body, ctype, {"Content-Disposition": f'attachment; filename="{mid}_{fname}"'})

        self._json({"error": "not found"}, 404)

    def _ask(self, mid: str):
        d = meeting_dir(mid)
        if not d:
            return self._json({"error": "meeting not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 16384:
                return self._json({"error": "Request body must be a small JSON object."}, 400)
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError
        except (ValueError, UnicodeDecodeError):
            return self._json({"error": "Request body must be JSON: {\"question\": \"...\"}"}, 400)
        try:
            refined, doc = ask_mod.load_meeting(d)
            return self._json(ask_mod.ask(body.get("question"), refined, doc))
        except ask_mod.AskError as e:
            return self._json({"error": str(e)}, e.status)

    def do_POST(self):
        path = urlparse(self.path).path
        
        # RETRY ENDPOINT
        m = re.fullmatch(r"/api/retry/([^/]+)", path)
        if m:
            mid = m.group(1)
            if not UPLOADS.is_dir():
                return self._json({"error": "Uploads directory not found"}, 404)
            # Find the original uploaded audio file safely
            audio = next((f for f in UPLOADS.iterdir() if f.stem == mid and f.suffix.lower() in AUDIO_EXT), None)
            if not audio:
                return self._json({"error": "Original audio file not found. Cannot retry."}, 404)
            
            meta = _load(OUTPUTS / mid / "pipeline_meta.json") or {}
            job = pr.new_job(mid)
            pr.run_pipeline(job, audio, OUTPUTS / mid, language=meta.get("language_choice", "en"))
            return self._json({"job_id": job["id"], "meeting_id": mid})

        # ASK PULSE ENDPOINT (read-only over refined_output.json / documentation_output.json)
        m = re.fullmatch(r"/api/meetings/([^/]+)/ask", path)
        if m:
            return self._ask(m.group(1))

        if path != "/api/upload":
            return self._json({"error": "not found"}, 404)
            
        name = Path(self.headers.get("X-Filename", "meeting.wav")).name
        ext = Path(name).suffix.lower()
        if ext not in AUDIO_EXT:
            return self._json({"error": f"unsupported file type {ext!r}"}, 400)
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return self._json({"error": "empty upload"}, 400)
        language = self.headers.get("X-Language", "en")
        try:
            from stt.languages import is_auto, route_for_language
            if not is_auto(language):
                route_for_language(language)
        except UnsupportedLanguageError as e:
            return self._json({"error": str(e)}, 400)
        stem = re.sub(r"[^\w.\- ]", "_", Path(name).stem)
        mid, n = stem, 1
        while (OUTPUTS / mid).exists():
            n += 1; mid = f"{stem}_{n}"
        UPLOADS.mkdir(exist_ok=True)
        audio = UPLOADS / f"{mid}{ext}"
        remaining = length
        with audio.open("wb") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                f.write(chunk); remaining -= len(chunk)
        job = pr.new_job(mid)
        pr.run_pipeline(job, audio, OUTPUTS / mid, language=language)
        self._json({"job_id": job["id"], "meeting_id": mid})


def main():
    global OUTPUTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--outputs", default=str(OUTPUTS))
    a = ap.parse_args()
    OUTPUTS = Path(a.outputs).resolve()
    sys.path.insert(0, str(ROOT))
    print(f"Serving http://127.0.0.1:{a.port}  (outputs: {OUTPUTS})")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()