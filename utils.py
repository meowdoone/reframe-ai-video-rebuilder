from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


ProgressCallback = Callable[[int, str], None]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}


class PipelineError(RuntimeError):
    pass


def setup_logging(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    return logging.getLogger("reframe")


def emit_progress(
    callback: Optional[ProgressCallback], value: int, message: str
) -> None:
    if callback:
        callback(max(0, min(100, int(value))), message)


def run_command(
    command: Sequence[str], label: str, check: bool = True
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise PipelineError(f"{label}失败：{detail[-1800:]}")
    return result


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise PipelineError(f"未找到 {name}。")
    return path


def probe_video(path: Path) -> Dict[str, Any]:
    result = run_command(
        [
            require_binary("ffprobe"),
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        "读取视频信息",
    )
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    if not video:
        raise PipelineError(f"{path.name} 不包含可读取的视频轨。")
    audio = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        None,
    )
    duration = float(
        video.get("duration") or data.get("format", {}).get("duration") or 0
    )
    frame_rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    try:
        numerator, denominator = frame_rate.split("/", 1)
        fps = float(numerator) / max(float(denominator), 1.0)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    return {
        "duration": round(duration, 3),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": round(fps, 3),
        "video_codec": video.get("codec_name"),
        "audio_codec": audio.get("codec_name") if audio else None,
        "has_audio": bool(audio),
        "size_bytes": int(data.get("format", {}).get("size") or path.stat().st_size),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_filename(value: str, fallback: str = "file") -> str:
    name = Path(value or "").name
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return sanitized[:120] or fallback


def redact_error(error: Exception, limit: int = 500) -> str:
    message = str(error)
    message = re.sub(r"sk-[A-Za-z0-9_*.-]+", "[redacted]", message)
    message = re.sub(
        r"(?i)(api[_ -]?key[\"'=:\s]+)[^\s,}\]]+",
        r"\1[redacted]",
        message,
    )
    return message[:limit]


def discover_video_files(
    inputs: Iterable[Path], recursive: bool = False
) -> List[Path]:
    discovered: List[Path] = []
    for raw_path in inputs:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            discovered.append(path)
            continue
        if not path.is_dir():
            continue
        iterator = path.rglob("*") if recursive else path.glob("*")
        discovered.extend(
            candidate.resolve()
            for candidate in iterator
            if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS
        )
    return sorted(dict.fromkeys(discovered))


def cleanup_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        candidate = Path(path)
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)
        else:
            candidate.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
