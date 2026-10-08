# Meeting Assistant

Turns a meeting recording into a speaker-labelled transcript, a cleaned-up transcript, and structured meeting documentation (summary, minutes, decisions, action items), with a small web UI (PULSE) to browse the results.

```
AUDIO
  -> STT             Deepgram Nova-3 (cloud API)           -> timestamped raw transcript
  -> Diarization     pyannote community-1 (runs locally)   -> who spoke when
  -> Alignment       merge the two                         -> final_output.json       (speaker-labelled "turns")
  -> LLM1            transcript refinement (DeepSeek)      -> refined_output.json     ("segments")
  -> Transformation  topics, decision/action candidates,
                     consensus + decision context          -> transformation_output.json
  -> LLM2            meeting documentation (DeepSeek)      -> documentation_output.json / .txt
```

The raw transcript is always preserved separately from the refined one. Documentation is generated from the refined, evidence-linked transcript, and decisions and action items are linked to the transcript segments that support them.

**Multilingual:** the meeting language is chosen at upload (English or Hindi). Non-English transcripts are translated to English and refined in one DeepSeek step, so all later stages work on English text while the original-language text is kept for audit. See [Multilingual meetings](#multilingual-meetings).

---

## Quick start

**Prerequisites:** Python 3.12 (3.10 / 3.11 also work) and ffmpeg (includes ffprobe).

| OS | Install ffmpeg |
|---|---|
| Ubuntu / Debian / WSL | `sudo apt install ffmpeg python3-venv` |
| macOS | `brew install ffmpeg` |
| Windows | handled by `setup_windows.ps1` (installs FFmpeg 7.1 shared via winget) |

**1. Run the setup script** (creates `.venv`, installs pinned dependencies, creates `.env`, runs the offline tests):

```bash
# Linux / macOS / WSL / Git-Bash
./setup.sh
# Windows PowerShell (needs Python 3.12 via the `py` launcher; NVIDIA GPU build of PyTorch)
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

The first run downloads PyTorch (a few GB), so it takes a few minutes. Re-running is safe.

**2. Add your API keys** to the `.env` created by setup (see [API keys](#api-keys)).

**3. Start the app:**

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m app.server               # open http://127.0.0.1:8000
```

Demo meetings are included in `outputs/`, so the UI shows data without any API key. Keys are only needed to process new audio.

<details>
<summary>Manual setup (without the scripts)</summary>

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
pytest -q                          # all tests should pass
```

For an NVIDIA GPU, install the CUDA build of PyTorch before `requirements.txt` (selector at pytorch.org; `torch==2.8.0`, `torchaudio==2.8.0`). Diarization also runs on CPU, just slower.
</details>

## API keys

Edit `.env` (git-ignored; never commit it). `.env.example` documents every setting.

| Variable | Needed for | Where to get it |
|---|---|---|
| `DEEPGRAM_API_KEY` | STT | console.deepgram.com |
| `HF_TOKEN` | pyannote diarization model download | Hugging Face read token; accept the terms for `pyannote/speaker-diarization-community-1` |
| `DEEPSEEK_API_KEY` | LLM1 refinement/translation, LLM2 documentation, Ask PULSE | platform.deepseek.com |

## Using it

### A. Web UI (recommended)

```bash
python -m app.server [--port 8000] [--outputs outputs]
```

- Lists every `outputs/<name>/` folder with Overview, Transcript, Analytics, Documentation and Settings views.
- **Upload audio** runs the whole chain with live stage status. Uploaded files are stored in `uploads/`.
- **Transcript tab:** *Raw* and *Refined* views (changes highlighted). For translated meetings: *Raw (Hindi)* and *Refined (English)*.
- **Evidence:** LLM2 `segment_ids` are resolved to real transcript segments by `app/evidence.py`; nothing is invented. Decisions also show their consensus level and the segments that explain it.

#### Ask PULSE

Click **Ask PULSE** to ask questions about the selected meeting, e.g. "What did we decide about PostgreSQL?" or "What is still unresolved?".

- Answers use only that meeting's `refined_output.json` and `documentation_output.json` (read-only).
- Every answer lists evidence cards (segment id, speaker, time, exact text); clicking one opens that segment in the Transcript tab.
- If the meeting does not contain the answer you get: "I couldn't find enough evidence in this meeting to answer that."
- API: `POST /api/meetings/<meeting_id>/ask` with `{"question": "..."}`. Response: `answer`, `found`, `confidence`, `evidence[]`. Errors: 400 invalid question, 404 unknown meeting, 409 no refined transcript yet, 502/503 model unavailable.
- Grounding: keyword retrieval picks the relevant segments, the model returns strict JSON with segment ids only, the backend drops any id that was not sent to the model and fills speaker/time/text itself, and an answer without a valid citation is withheld. Answers can still be wrong, so check the cited evidence.

### B. Command line, stage by stage

Run from the project root with the venv active. Each stage reads the previous stage's folder.

```bash
python -m tests.run_pipeline meeting.mp3 [--language en] [--num-speakers 3] [--outdir outputs]
        # STT + diarization + alignment -> outputs/meeting/{raw_transcript.txt, final_output.json}
python -m tests.run_llm1            outputs/meeting   # -> refined_output.json
python -m tests.run_transformation  outputs/meeting   # -> transformation_output.json
python -m tests.run_llm2            outputs/meeting   # -> documentation_output.json / .txt
```

Single-stage helpers: `python -m tests.run_stt meeting.mp3 --out raw_stt.json` and `python -m tests.run_diarization meeting.mp3`.

`--language`: `en` (default), `hi`, `multi` (code-switching) or `detect`.

### C. Run from an existing transcript (skip STT)

If you already have a speaker-labelled transcript, put it at `outputs/<name>/final_output.json` and run:

```bash
python tests/run_from_transcript.py outputs/<name>                      # English
python tests/run_from_transcript.py outputs/<name> --language hi-IN     # Hindi
```

Required format:

```json
{ "turns": [ { "index": 0, "speaker": "Speaker 01", "start": 0.0, "end": 7.1, "text": "..." } ] }
```

`speaker`, `start`, `end` (seconds, in order) and `text` are required; `index` is optional. This runs LLM1, Transformation and LLM2 and writes all outputs into the same folder.

### D. Tests

```bash
pytest -q          # fully offline: no API keys, no network, no GPU needed
```

## Project layout

```
stt/              STT providers (Deepgram Nova-3 default)
diarization/      pyannote wrapper + post-processing
alignment/        merges STT words with diarization into speaker turns
llm1/             transcript refinement and translation (DeepSeek)
transformation/   evidence-grounded structure: topics, decision/action candidates, consensus, context
llm2/             meeting documentation generation + guidance cross-check
app/              web UI server (stdlib only) + static frontend; app/ask.py = Ask PULSE
common/           audio validation (ffprobe/ffmpeg) and shared errors
tests/            unit tests (test_*.py) and runnable stage scripts (run_*.py)
outputs/          one folder per processed meeting (git-ignored except demo meetings)
setup.sh / setup_windows.ps1   reproducible environment setup
```

## Pipeline contracts

- `STTProvider.transcribe(path)` -> `STTResult`
- `PyannoteDiarizer.diarize(path)` -> `DiarizationResult`
- `align_transcript(STTResult, DiarizationResult)` -> `SpeakerLabelledTranscript`
- `refine_file`, `transform_file`, `document_file`: each reads the JSON artifact of the previous stage.

Stages communicate through frozen Pydantic models and JSON artifacts, so each can be run and tested independently. External model calls are isolated inside their stage; raw, refined and documented outputs are all persisted for reproducibility.

**Data contract:** `final_output.json` uses `turns`; `refined_output.json` uses `segments`. This is intentional. Segment IDs are positions in the list and are shared by every later stage.

## How it works

- **Audio** is validated with ffprobe (missing / empty / unreadable / no audio stream / too short -> typed errors), then converted to 16 kHz mono FLAC for upload.
- **Deepgram:** `POST /v1/listen?model=nova-3&smart_format=true&utterances=true`; retries on 429/5xx/network errors with backoff. Deepgram's own diarization is off; speakers come from pyannote.
- **Alignment:** per-word max-overlap speaker vote; segments split where the speaker changes; overlap flagged (`has_overlap`); words in diarization gaps take the nearest speaker within 0.6 s else `UNKNOWN`; short weakly supported runs are smoothed into neighbours; same-speaker turns < 1 s apart are merged. Tunables: `AlignConfig` in `alignment/align.py`. Diarization post-processing is tunable via the `DIARIZATION_*` settings in `.env`.
- **LLM1** corrects transcription errors conservatively (technical terms, names), logs every change, and flags possible negation or uncertainty loss.
- **LLM2** writes the documentation from the refined transcript. The Transformation output is only guidance; LLM2 must verify every candidate against the transcript, and a deterministic cross-check warns when they disagree.

### Consensus and decision context (Transformation)

Each decision candidate carries two additional, optional layers. Neither changes the candidate's `status`.

- **Consensus:** a deterministic, rule-based analysis of the cited segments: explicit agreement, finalization, supporting statements, opposition, and unresolved ("still open") language, evaluated per clause so that "X is decided, Y is still open" does not make X unresolved. Output: a classification (`confirmed`, `likely`, `proposed`, `unresolved`, `contested`), a level (high/medium/low), the segment IDs that caused it, and a heuristic score. The score is an internal confidence, **not** a calibrated probability. Silence is never treated as agreement.
- **Context reconstruction:** a bounded set of surrounding segments that explains the decision (e.g. a final "let's go with that" plus the proposal it refers to) and a self-contained `reconstructed_text`. Model-written wording is accepted only if it passes grounding checks (numbers, identifiers, negation, uncertainty); otherwise a fallback built from existing text is used.

## Multilingual meetings

English and Hindi meetings follow the same pipeline. For a non-English meeting, `final_output.json` holds the original-language turns plus language metadata, and LLM1 (`llm1/translation.py`) translates and refines them to English in one DeepSeek step:

```
Audio -> STT -> diarization + alignment -> final_output.json   (original language + language metadata)
      -> LLM1 translation + refinement  -> refined_output.json (original_text + English refined_text)
      -> Transformation + evidence      -> LLM2 documentation  (written from the English text)
```

- **Provenance:** each refined segment keeps `original_text` (source language), `refined_text` (English) and `source_language`, plus speaker, timestamps and index. Evidence citations resolve to the original segment (SOURCE vs FINAL in the evidence drawer).
- **UI:** the Transcript tab offers *Raw (Hindi)* and *Refined (English)*; the *Original-language audit* download is also available.
- **Tests:** `pytest tests/test_multilingual.py` (all external model calls are mocked).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `'ffprobe' not found on PATH` | Install ffmpeg (see prerequisites) and reopen the terminal. |
| `setup.sh`: "Need Python 3.10, 3.11 or 3.12" | The pinned stack (PyTorch 2.8, pyannote 4.0.7) is tested on Python 3.12. Install Python 3.12. |
| `ensurepip` / venv creation fails on Ubuntu | `sudo apt install python3-venv` |
| Hugging Face 401/403 during diarization | Accept the model terms (link above) with the account that owns `HF_TOKEN`. |
| torchcodec / FFmpeg DLL errors on Windows | Use `setup_windows.ps1` (installs FFmpeg 7.1 shared and copies the DLLs next to TorchCodec). |
| CUDA out of memory | Set `DIARIZATION_DEVICE=cpu` in `.env`. |
| `DEEPSEEK_API_KEY is not set` | Add your DeepSeek key to `.env`. LLM1, LLM2 and Ask PULSE all use it. |
| UI is empty | No folder in `outputs/` yet; upload audio or run the CLI pipeline. |
| Transcript tab empty after an update | Hard-refresh the browser (Ctrl+F5); an old copy of the page may be cached. |

## Known caveats

- Live-API behaviour (Deepgram `language=detect`, keyterm prompting) follows the providers' documentation and has not been exercised exhaustively.
- Deepgram does not store transcripts; the saved JSON in `outputs/` is your only copy.
- LLM output can be wrong. Decisions and action items are guidance with evidence links; check the cited segments before relying on them.
- Action-item owners are only filled in when the meeting names a person or role explicitly; a speaker label alone is not treated as a named owner.
