from __future__ import annotations
import json, shutil, subprocess
from dataclasses import dataclass
from pathlib import Path
from .errors import (AudioNotFoundError, EmptyAudioError,
                     UnreadableAudioError, DependencyError)

MIN_DURATION_S = 0.1


@dataclass(frozen=True)
class AudioInfo:
    path: Path
    duration: float
    size_bytes: int


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        raise DependencyError(f"'{tool}' not found on PATH. Install ffmpeg (it ships ffprobe).")


def _run(cmd: list[str], what: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()[-300:]
        raise UnreadableAudioError(f"ffmpeg failed while {what}: {err}")
    return proc


def validate_audio(path: str | Path) -> AudioInfo:
    """Raises AudioNotFoundError / EmptyAudioError / UnreadableAudioError / DependencyError."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise AudioNotFoundError(f"Audio file not found: {p}")
    size = p.stat().st_size
    if size == 0:
        raise EmptyAudioError(f"Audio file is empty (0 bytes): {p}")
    _require("ffprobe"); _require("ffmpeg")
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type:format=duration",
         "-of", "json", str(p)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise UnreadableAudioError(f"Cannot read {p.name}: {proc.stderr.strip()[-300:]}")
    info = json.loads(proc.stdout or "{}")
    if not info.get("streams"):
        raise UnreadableAudioError(f"No audio stream found in {p.name}")
    try:
        duration = float(info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        # some containers (e.g. browser webm) carry no duration: measure by decoding
        out = _run(["ffmpeg", "-v", "error", "-i", str(p), "-map", "0:a:0",
                    "-ar", "16000", "-ac", "1", "-f", "s16le", "-"], "measuring duration")
        duration = len(out.stdout) / 32000.0
    if duration < MIN_DURATION_S:
        raise EmptyAudioError(f"Audio is too short ({duration:.3f}s): {p.name}")
    return AudioInfo(path=p, duration=duration, size_bytes=size)


def to_16k_mono(src: Path, dst: Path, codec: str = "flac") -> None:
    """codec: 'flac' for upload, 'pcm_s16le' for wav."""
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-map", "0:a:0",
          "-ar", "16000", "-ac", "1", "-c:a", codec, str(dst)], "converting audio")


def extract_chunk(src: Path, dst: Path, start: float, length: float) -> None:
    _run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
          "-i", str(src), "-c:a", "flac", str(dst)], "cutting chunk")


def plan_chunks(duration: float, chunk_s: float, min_tail: float = 5.0) -> list[tuple[float, float]]:
    """(start, length) pairs. A tiny tail is folded into the previous chunk."""
    if duration <= chunk_s + min_tail:
        return [(0.0, duration)]
    starts, t = [], 0.0
    while t < duration:
        starts.append(t)
        t += chunk_s
    if duration - starts[-1] < min_tail and len(starts) > 1:
        starts.pop()
    return [(s, (starts[i + 1] if i + 1 < len(starts) else duration) - s) for i, s in enumerate(starts)]
