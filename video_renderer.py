from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from config import RenderSettings
from utils import PipelineError, probe_video, require_binary, run_command


@dataclass
class RenderResult:
    output_path: Path
    used_narration: bool
    metadata: Dict[str, Any]
    filter_graph: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_file": self.output_path.name,
            "used_narration": self.used_narration,
            "filter_graph": self.filter_graph,
            **self.metadata,
        }


def _validate_settings(settings: RenderSettings) -> None:
    if settings.width <= 0 or settings.height <= 0:
        raise PipelineError("输出宽高必须大于 0。")
    if settings.width % 2 or settings.height % 2:
        raise PipelineError("H.264 输出宽高必须是偶数。")
    if not 0.5 <= settings.crop_scale <= 1.0:
        raise PipelineError("crop_scale 必须在 0.5 到 1.0 之间。")
    if not 1 <= settings.fps <= 60:
        raise PipelineError("输出帧率必须在 1 到 60 之间。")


def build_video_filter(settings: RenderSettings) -> str:
    _validate_settings(settings)
    crop_scale = f"{settings.crop_scale:.5f}"
    return (
        f"crop=trunc(iw*{crop_scale}/2)*2:trunc(ih*{crop_scale}/2)*2:"
        "(iw-ow)/2:(ih-oh)/2,"
        f"scale={settings.width}:{settings.height}:"
        "force_original_aspect_ratio=increase,"
        f"crop={settings.width}:{settings.height},"
        f"fps={settings.fps},setsar=1,"
        f"eq=contrast={settings.contrast:.4f}:"
        f"saturation={settings.saturation:.4f}:"
        f"brightness={settings.brightness:.4f},"
        "format=yuv420p"
    )


def render_video(
    input_video: Path,
    narration_audio: Optional[Path],
    output_path: Path,
    settings: RenderSettings,
) -> RenderResult:
    input_video = Path(input_video).resolve()
    output_path = Path(output_path).resolve()
    if not input_video.exists():
        raise PipelineError(f"输入视频不存在：{input_video}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_filter = build_video_filter(settings)
    ffmpeg = require_binary("ffmpeg")
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_video),
    ]
    used_narration = bool(narration_audio and Path(narration_audio).exists())
    if used_narration:
        command.extend(["-i", str(Path(narration_audio).resolve())])
        filter_graph = (
            f"[0:v:0]{video_filter}[v];"
            "[1:a:0]aresample=48000,"
            "loudnorm=I=-14:TP=-1.5:LRA=11[a]"
        )
    else:
        command.extend(
            [
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000",
            ]
        )
        filter_graph = f"[0:v:0]{video_filter}[v];[1:a:0]anull[a]"
    command.extend(
        [
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-map_metadata",
            "-1",
            "-c:v",
            "libx264",
            "-preset",
            settings.preset,
            "-crf",
            str(settings.crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-metadata:s:v:0",
            f"handler_name={settings.handler_video}",
            "-metadata:s:a:0",
            f"handler_name={settings.handler_audio}",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ]
    )
    run_command(command, "视频画音重构")
    metadata = probe_video(output_path)
    if (metadata["width"], metadata["height"]) != (
        settings.width,
        settings.height,
    ):
        raise PipelineError("成片尺寸与配置不一致。")
    return RenderResult(
        output_path=output_path,
        used_narration=used_narration,
        metadata=metadata,
        filter_graph=video_filter,
    )
