from __future__ import annotations
import os, tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from common.audio import validate_audio, to_16k_mono
from common.errors import DiarizationError
from .postprocess import PostConfig, Seg, postprocess
from .schemas import DiarizationResult, DiarSegment


@dataclass
class RawDiarization:
    """Un-post-processed pipeline output (plain tuples) - used by diarize() and the diagnostics."""
    full: list = field(default_factory=list)        # regular diarization (may overlap)
    exclusive: list = field(default_factory=list)   # exclusive diarization ([] if not provided)
    centroids: Optional[dict] = None                # {speaker: embedding list} if the pipeline returned them
    warnings: list = field(default_factory=list)
    device: str = "cpu"
    out_fields: list = field(default_factory=list)


class PyannoteDiarizer:
    MODEL_ID = "pyannote/speaker-diarization-community-1"

    def __init__(self, model: str = MODEL_ID, hf_token: Optional[str] = None,
                 device: Optional[str] = None, postprocess: Optional[bool] = None,
                 post_config: Optional[PostConfig] = None, cluster_threshold: Optional[float] = None):
        self.model = model
        self.hf_token = hf_token
        self.device = device or os.environ.get("DIARIZATION_DEVICE", "cpu")  # cpu | cuda | auto
        if postprocess is None:
            postprocess = os.environ.get("DIARIZATION_POSTPROCESS", "1").strip().lower() not in ("0", "false", "no", "")
        self.postprocess = postprocess
        self.post_config = post_config or PostConfig.from_env()
        if cluster_threshold is None and os.environ.get("DIARIZATION_CLUSTER_THRESHOLD", "").strip():
            cluster_threshold = float(os.environ["DIARIZATION_CLUSTER_THRESHOLD"])
        self.cluster_threshold = cluster_threshold
        self._pipeline = None
        self._device_used = "cpu"

    # ---- pipeline loading / tuning ------------------------------------
    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        token = self.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise DiarizationError("HF_TOKEN not set. Accept the model terms on Hugging Face and create a token.")
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise DiarizationError("pyannote.audio/torch not installed (pip install pyannote.audio).") from e
        try:
            try:
                pipe = Pipeline.from_pretrained(self.model, token=token)
            except TypeError:  # older pyannote
                pipe = Pipeline.from_pretrained(self.model, use_auth_token=token)
        except Exception as e:
            raise DiarizationError(f"Could not load {self.model}: {e}") from e
        if pipe is None:
            raise DiarizationError(f"Could not load {self.model}: check HF_TOKEN and that you accepted the model terms.")
        dev = self.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(torch.device(dev))
        self._device_used = dev
        if self.cluster_threshold is not None:
            self._set_threshold(pipe, self.cluster_threshold)
        self._pipeline = pipe
        return pipe

    @staticmethod
    def _set_threshold(pipe, value: float) -> None:
        try:
            params = pipe.parameters(instantiated=True)
            clustering = params.get("clustering")
            if not isinstance(clustering, dict) or "threshold" not in clustering:
                raise KeyError(f"no clustering.threshold among parameters {list(params)}")
            clustering["threshold"] = float(value)
            pipe.instantiate(params)
        except Exception as e:
            raise DiarizationError(f"Could not set clustering.threshold={value}: {e!r}") from e

    def set_cluster_threshold(self, value: float) -> None:
        self._set_threshold(self._load(), value)
        self.cluster_threshold = value

    def pipeline_params(self) -> dict:
        try:
            return self._load().parameters(instantiated=True)
        except DiarizationError:
            raise
        except Exception as e:
            return {"error": repr(e)}

    # ---- inference ------------------------------------------------------
    @staticmethod
    def _centroids(out, full, warnings: list) -> Optional[dict]:
        emb = getattr(out, "speaker_embeddings", None)
        if emb is None:
            return None
        try:
            import numpy as np
            arr = np.asarray(emb, dtype="float64")
            labels = list(full.labels())   # assumption: embedding rows follow the label order
            if arr.ndim != 2 or arr.shape[0] != len(labels):
                warnings.append(f"speaker_embeddings shape {arr.shape} != {len(labels)} speakers; ignored")
                return None
            return {str(l): arr[i].tolist() for i, l in enumerate(labels) if np.isfinite(arr[i]).all()} or None
        except Exception as e:
            warnings.append(f"could not read speaker_embeddings: {e!r}")
            return None

    def infer_raw(self, audio_path: str | Path, num_speakers: Optional[int] = None,
                  min_speakers: Optional[int] = None, max_speakers: Optional[int] = None) -> RawDiarization:
        info = validate_audio(audio_path)
        pipe = self._load()
        import numpy as np, soundfile as sf, torch
        with tempfile.TemporaryDirectory(prefix="diar_") as td:
            wav = Path(td) / "audio.wav"
            to_16k_mono(info.path, wav, "pcm_s16le")
            data, sr = sf.read(str(wav), dtype="float32", always_2d=True)  # (time, ch)
        # Pass a waveform so pyannote needs no audio decoder of its own
        audio = {"waveform": torch.from_numpy(np.ascontiguousarray(data.T)), "sample_rate": sr}
        kwargs = {k: v for k, v in dict(num_speakers=num_speakers, min_speakers=min_speakers,
                                        max_speakers=max_speakers).items() if v is not None}
        warnings: list[str] = []
        try:
            try:
                out = pipe(audio, **kwargs)
            except torch.cuda.OutOfMemoryError:
                warnings.append("CUDA out of memory; retried on CPU.")
                pipe.to(torch.device("cpu")); self._device_used = "cpu"
                torch.cuda.empty_cache()
                out = pipe(audio, **kwargs)
        except Exception as e:
            raise DiarizationError(f"Diarization failed: {e}") from e

        full = getattr(out, "speaker_diarization", out)
        excl = getattr(out, "exclusive_speaker_diarization", None)

        def to_segs(ann) -> list:
            if ann is None:
                return []
            segs = [(round(float(t.start), 3), round(float(t.end), 3), str(lab))
                    for t, _, lab in ann.itertracks(yield_label=True) if t.end > t.start]
            return sorted(segs, key=lambda s: (s[0], s[1]))

        if excl is None:
            warnings.append("pipeline output has no exclusive_speaker_diarization; "
                            "alignment will fall back to the regular diarization.")
        try:
            fields = [a for a in dir(out) if not a.startswith("_") and not callable(getattr(out, a, None))]
        except Exception:
            fields = []
        return RawDiarization(full=to_segs(full), exclusive=to_segs(excl),
                              centroids=self._centroids(out, full, warnings),
                              warnings=warnings, device=self._device_used, out_fields=fields)

    def diarize(self, audio_path: str | Path, num_speakers: Optional[int] = None,
                min_speakers: Optional[int] = None, max_speakers: Optional[int] = None) -> DiarizationResult:
        raw = self.infer_raw(audio_path, num_speakers, min_speakers, max_speakers)
        full, excl = raw.full, raw.exclusive
        warnings = list(raw.warnings)
        if self.postprocess and full:
            pr = postprocess(full, excl, raw.centroids, self.post_config)
            full, excl = pr.full, pr.exclusive
            warnings += [f"postprocess: {n}" for n in pr.notes]
        if not full:
            warnings.append("No speech/speakers detected by diarization.")
        seg = lambda xs: tuple(DiarSegment(start=s, end=e, speaker=k) for s, e, k in xs)
        return DiarizationResult(segments=seg(full), exclusive_segments=seg(excl),
                                 speakers=tuple(sorted({k for _, _, k in full})),
                                 model=self.model, device=raw.device,
                                 audio_path=str(Path(audio_path)), warnings=tuple(warnings))
