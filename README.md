# Meeting Assistant

Turns a meeting recording into a speaker-labelled transcript, a cleaned-up transcript, and structured meeting documentation (summary, minutes, decisions, action items), with a small web UI to browse the results.

```
AUDIO
  -> STT            Deepgram Nova-3 (cloud API)          -> timestamped raw transcript
  -> Diarization    pyannote community-1 (runs locally)  -> who spoke when
  -> Alignment      merge the two                        -> final_output.json  (speaker-labelled)
  -> LLM1           transcript refinement (DeepSeek)     -> refined_output.json
  -> Transformation topics / decision & action candidates -> transformation_output.json
  -> LLM2           meeting documentation                -> documentation_output.json / .txt
```

**Multilingual:** The meeting language is selected at upload. The system supports English and Hindi recordings, which pass through Deepgram Nova-3 -> diarization -> alignment -> LLM1 transcript refinement -> transformation and evidence processing -> LLM2 documentation. The raw transcript is preserved separately from the refined transcript, with the final documentation generated from the refined, evidence-linked representation. See [Multilingual meetings](#multilingual-meetings).

## Quick start (3 steps)

**Prerequisites:** Python 3.12 (3.10 / 3.11 also work) and `ffmpeg` (includes `ffprobe`).

| OS | Install ffmpeg |
|---|---|
| Ubuntu / Debian / WSL | `sudo apt install ffmpeg python3-venv` |
| macOS | `brew install ffmpeg` |
| Windows | handled by `setup_windows.ps1` (installs FFmpeg 7.1 shared via winget) |

**1. Run the setup script** (creates `.venv`, installs pinned dependencies, creates `.env`, runs the offline tests):

```bash
# Linux / macOS / WSL / Git-Bash
./setup.sh
```
```powershell
# Windows PowerShell (needs Python 3.12 via the `py` launcher; NVIDIA GPU build of PyTorch)
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

The first run downloads PyTorch (a few GB), so it takes a few minutes. Re-running is safe.

**2. Add your API keys** to the `.env` file that setup created (see [API keys](#api-keys)).

**3. Start the app:**

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m app.server               # open http://127.0.0.1:8000
```

Two demo meetings (`outputs/test2.wav`, `outputs/Bdb001_30min`) are included, so the UI shows data **without any API key**. Keys are only needed to process new audio.

### Manual setup (if you prefer not to use the scripts)

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
pytest -q                          # should print: 181 passed
```

Optional: for an NVIDIA GPU on Linux/Windows, install the CUDA build of PyTorch **before** `requirements.txt`
(selector at pytorch.org; `torch==2.8.0`, `torchaudio==2.8.0`). Diarization runs fine on CPU, just slower.

## API keys

Edit `.env` (it is git-ignored; never commit it). `.env.example` documents every required setting.

| Variable | Needed for | Where to get it |
|---|---|---|
| `DEEPGRAM_API_KEY` | STT | console.deepgram.com |
| `HF_TOKEN` | PyAnnote diarization model download | Hugging Face read token. Accept the terms for `pyannote/speaker-diarization-community-1` |
| `DEEPSEEK_API_KEY` | LLM1 transcript refinement and LLM2 documentation | platform.deepseek.com |
## Using it

### A. Web UI (recommended)

```bash
python -m app.server [--port 8000] [--outputs outputs]
```

- Lists every `outputs/<name>/` folder and shows raw, refined and documentation views.
- **Upload audio** runs the whole chain (STT -> diarization -> alignment -> LLM1 -> transformation -> LLM2) with live stage status. Uploaded files are stored in `uploads/`.
- LLM2 `segment_ids` are resolved to the real transcript segments by `app/evidence.py`; nothing is invented.

### Ask PULSE (questions about a meeting)

Click **Ask PULSE** in the header to open a chat panel for the selected meeting, e.g. *"What did we decide about PostgreSQL?"*, *"What action items were assigned?"*, *"What is still unresolved?"*.

- Answers are built **only** from that meeting's `refined_output.json` + `documentation_output.json` (read-only; nothing is written).
- Every answer lists **evidence cards** (segment id, speaker, time, exact transcript text). Clicking a card opens and highlights that segment in the Transcript tab (the same navigation used by the Documentation view).
- If the meeting does not contain the answer you get: *"I couldn't find enough evidence in this meeting to answer that."* (plus any related passages found).
- It uses the same LLM client and keys as the pipeline (`LLM1_PROVIDER` / `DEEPSEEK_API_KEY` in `.env`). Answers can still be wrong: check the cited evidence.
- API: `POST /api/meetings/<meeting_id>/ask` with `{"question": "..."}`. The response contains `answer`, `found`, `confidence`, and `evidence[]` (`segment_id`, `speaker`, `start`, `end`, `text`, `original_text`, `changed`). Errors: 400 empty/invalid question, 404 unknown meeting, 409 no refined transcript yet, 502/503 model unavailable.
- How grounding works: keyword/phrase retrieval picks the relevant segments (plus decision/action segments for those question types) -> the model returns strict JSON with segment ids only -> the backend drops every id that is not in the meeting or was not sent to the model, and fills speaker/time/text itself from the meeting data -> an answer without a valid citation is withheld.
- Tests: `pytest tests/test_ask.py -q` (offline, uses a scripted fake model).

### B. Command line, stage by stage

Run from the project root with the venv active. Each stage reads the previous stage's output folder.

```bash
python -m tests.run_pipeline meeting.mp3 [--language en] [--num-speakers 3] [--outdir outputs]
        # STT + diarization + alignment -> outputs/meeting/{raw_transcript.txt, final_output.json}
python -m tests.run_llm1            outputs/meeting   # -> refined_output.json
python -m tests.run_transformation  outputs/meeting   # -> transformation_output.json
python -m tests.run_llm2            outputs/meeting   # -> documentation_output.json / .txt
```

Single-stage helpers:

```bash
python -m tests.run_stt meeting.mp3 --out raw_stt.json     # STT only
python -m tests.run_diarization meeting.mp3                # diarization only
```

Language: `--language en` (default), another code (e.g. `hi`), `multi` (code-switching) or `detect`.

### C. Tests

```bash
pytest -q          # fully offline: no API keys, no network, no GPU needed
```

## Project layout

```
stt/              STT providers (Deepgram Nova-3 default; Groq Whisper optional)
diarization/      pyannote wrapper + post-processing
alignment/        merges STT words with diarization into speaker turns
llm1/             transcript refinement (DeepSeek; Groq/Gemini/Cerebras fallbacks)
transformation/   evidence-grounded structure (topics, decision/action candidates)
llm2/             meeting documentation generation
app/              web UI server (stdlib only) + static frontend; app/ask.py = Ask PULSE service
common/           audio validation (ffprobe/ffmpeg) and shared errors
tests/            unit tests (test_*.py) and runnable stage scripts (run_*.py)
outputs/          one folder per processed meeting (git-ignored except the demo meetings)
setup.sh / setup_windows.ps1   reproducible environment setup
```

## Pipeline Contracts

- `STTProvider.transcribe(path)` → `STTResult` (`stt/base.py`, schemas in `stt/schemas.py`)
- `PyannoteDiarizer.diarize(path)` → `DiarizationResult`
- `align_transcript(STTResult, DiarizationResult)` → `SpeakerLabelledTranscript`
- `refine_file`, `transform_file`, `document_file`: each operates on the JSON artifacts produced by the preceding stage

The pipeline stages communicate through well-defined Pydantic data models and JSON artifacts. This keeps the individual components modular while allowing each stage to be tested and executed independently.

The current STT implementation uses **Deepgram Nova-3**. The pipeline preserves the raw transcript before refinement, with subsequent stages producing separate refined and documented outputs.

The pipeline uses frozen Pydantic models for its intermediate results, with `.model_dump()` and `.model_dump_json()` available for serialization. The stage boundaries are kept independent of any orchestration framework, making the existing pipeline suitable for integration with frameworks such as LangGraph without requiring changes to the underlying data contracts.

The current STT implementation uses **Deepgram Nova-3**. External model calls are isolated behind their respective pipeline stages, while the raw, refined, and documented outputs are persisted as JSON artifacts for reproducibility and downstream processing.
## How it works

- Audio is validated with ffprobe (missing / empty / unreadable / no audio stream / too short -> typed errors), then converted to 16 kHz mono FLAC for upload.
- Deepgram call: `POST /v1/listen?model=nova-3&smart_format=true&utterances=true&language=en`; retries on 429/5xx/network errors with backoff. Deepgram's own diarization is off; speakers come from pyannote.
- Alignment: per-word max-overlap speaker vote; segments split where the speaker changes; overlap is flagged (`has_overlap`), words in diarization gaps take the nearest speaker within 0.6 s else `UNKNOWN`; short weakly supported runs are smoothed into neighbours; same-speaker turns < 1 s apart are merged. Tunables: `AlignConfig` in `alignment/align.py`.
- Diarization post-processing (merge similar speakers, drop spurious ones) is tunable via the `DIARIZATION_*` settings in `.env`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `'ffprobe' not found on PATH` | Install ffmpeg (see prerequisites) and reopen the terminal. |
| `setup.sh`: "Need Python 3.10, 3.11 or 3.12" | The pinned stack (PyTorch 2.8, pyannote 4.0.7) is tested on Python 3.12. Install Python 3.12. |
| `ensurepip` / venv creation fails on Ubuntu | `sudo apt install python3-venv` |
| Hugging Face 401/403 during diarization | Accept the model terms (link above) with the same account that owns `HF_TOKEN`. |
| `torchcodec` / FFmpeg DLL errors on Windows | Use `setup_windows.ps1` (installs FFmpeg 7.1 shared and copies the DLLs next to TorchCodec). |
| CUDA out of memory | Set `DIARIZATION_DEVICE=cpu` in `.env`. |
| `DEEPSEEK_API_KEY is not set` | Fill it in `.env`, or set `LLM1_PROVIDER` to another provider you have a key for. |
| UI is empty | No folder in `outputs/` yet; upload audio or run the CLI pipeline. |

## Known caveats

- Live-API behaviour (Deepgram `language=detect`, `keyterm` prompting) follows the providers' documentation and has not been exercised exhaustively.
- Deepgram does not store transcripts; the saved JSON in `outputs/` is your only copy.
- Deepgram pricing varies between sources; confirm at deepgram.com/pricing.
- The UI cannot show per-item status (confirmed/proposed/pending): the pipeline JSON does not record it. Action-item status is always "Not stated".
- `tests/test.wav` is a small sample clip used for manual runs.


## Multilingual meetings

The meeting assistant supports both **English and Hindi** meeting recordings while keeping the downstream processing pipeline consistent across languages.

```text
Audio
  |
  +--> Deepgram Nova-3
  |        |
  |        +--> timestamped transcript
  |
  +--> PyAnnote Community-1
           |
           +--> diarization + alignment
                    |
                    +--> final_output.json
                    |      (original transcript + language metadata)
                    |
                    +--> LLM1: translation.py
                           |
                           +--> translation + transcript refinement
                           |
                           +--> refined_output.json
                                  |
                                  +--> Transformation + evidence
                                          |
                                          +--> LLM2
                                                 |
                                                 +--> documentation
* **Evidence / UI:** segment ids are positions (same as today), so LLM2 citations resolve to the original-language segment (SOURCE vs FINAL in the evidence drawer). The Transcript tab shows only *Translated & Refined Transcript*; the original text is available in evidence and via the *Original-language audit* download.
* **Tests:** `pytest tests/test_multilingual.py` (all Sarvam/DeepSeek/Deepgram calls are mocked).
