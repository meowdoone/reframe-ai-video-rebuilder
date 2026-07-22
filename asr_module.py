from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import ASRSettings
from utils import PipelineError, redact_error, require_binary, run_command


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class ASRResult:
    text: str
    segments: List[TranscriptSegment]
    mode: str
    language: Optional[str]
    audio_path: Optional[Path]
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "segments": [asdict(segment) for segment in self.segments],
            "mode": self.mode,
            "language": self.language,
            "audio_file": self.audio_path.name if self.audio_path else None,
            "warning": self.warning,
        }


_FASTER_MODELS: Dict[Tuple[str, str, str], Any] = {}
_WHISPER_MODELS: Dict[Tuple[str, str], Any] = {}


def extract_audio(video_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            require_binary("ffmpeg"),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        "提取音频",
    )
    return output_path


def _transcribe_faster(
    audio_path: Path, settings: ASRSettings
) -> Tuple[str, List[TranscriptSegment], Optional[str]]:
    from faster_whisper import WhisperModel

    key = (settings.model, settings.device, settings.compute_type)
    model = _FASTER_MODELS.get(key)
    if model is None:
        model = WhisperModel(
            settings.model,
            device=settings.device,
            compute_type=settings.compute_type,
        )
        _FASTER_MODELS[key] = model
    raw_segments, info = model.transcribe(
        str(audio_path),
        language=settings.language or None,
        vad_filter=True,
    )
    segments = [
        TranscriptSegment(
            start=round(float(segment.start), 3),
            end=round(float(segment.end), 3),
            text=str(segment.text).strip(),
        )
        for segment in raw_segments
        if str(segment.text).strip()
    ]
    return " ".join(segment.text for segment in segments), segments, info.language


def _transcribe_whisper(
    audio_path: Path, settings: ASRSettings
) -> Tuple[str, List[TranscriptSegment], Optional[str]]:
    import whisper

    device = settings.device if settings.device != "auto" else "auto"
    key = (settings.model, device)
    model = _WHISPER_MODELS.get(key)
    if model is None:
        load_kwargs: Dict[str, Any] = {}
        if settings.device != "auto":
            load_kwargs["device"] = settings.device
        model = whisper.load_model(settings.model, **load_kwargs)
        _WHISPER_MODELS[key] = model
    result = model.transcribe(
        str(audio_path),
        language=settings.language or None,
        fp16=False,
        verbose=False,
    )
    segments = [
        TranscriptSegment(
            start=round(float(segment.get("start") or 0), 3),
            end=round(float(segment.get("end") or 0), 3),
            text=str(segment.get("text") or "").strip(),
        )
        for segment in result.get("segments", [])
        if str(segment.get("text") or "").strip()
    ]
    text = str(result.get("text") or "").strip()
    if not text:
        text = " ".join(segment.text for segment in segments)
    return text, segments, result.get("language")


def _fallback_result(
    settings: ASRSettings,
    audio_path: Optional[Path],
    warning: str,
) -> ASRResult:
    default_text = settings.default_text.strip()
    return ASRResult(
        text=default_text,
        segments=[],
        mode="default-text" if default_text else "skipped",
        language=settings.language or None,
        audio_path=audio_path,
        warning=warning,
    )


def transcribe_video(
    video_path: Path,
    work_dir: Path,
    settings: ASRSettings,
) -> ASRResult:
    video_path = Path(video_path).resolve()
    work_dir = Path(work_dir).resolve()
    audio_path = work_dir / "source_audio.wav"
    try:
        extract_audio(video_path, audio_path)
    except Exception as exc:
        return _fallback_result(settings, None, redact_error(exc))

    backend = settings.backend.lower().strip()
    if backend not in {"auto", "faster-whisper", "whisper"}:
        raise PipelineError(f"不支持的 ASR 后端：{settings.backend}")
    candidates = (
        ["faster-whisper", "whisper"]
        if backend == "auto"
        else [backend]
    )
    errors: List[str] = []
    for candidate in candidates:
        try:
            if candidate == "faster-whisper":
                text, segments, language = _transcribe_faster(audio_path, settings)
            else:
                text, segments, language = _transcribe_whisper(audio_path, settings)
            if text.strip():
                return ASRResult(
                    text=text.strip(),
                    segments=segments,
                    mode=candidate,
                    language=language or settings.language or None,
                    audio_path=audio_path,
                )
            errors.append(f"{candidate}: 未识别到文本")
        except Exception as exc:
            errors.append(f"{candidate}: {redact_error(exc, 240)}")
    return _fallback_result(settings, audio_path, "；".join(errors))
