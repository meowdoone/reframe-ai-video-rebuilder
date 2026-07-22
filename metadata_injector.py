from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from config import MetadataSettings
from utils import redact_error, run_command


@dataclass
class MetadataResult:
    applied: bool
    mode: str
    requested: Dict[str, str]
    readback: Dict[str, Any]
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applied": self.applied,
            "mode": self.mode,
            "requested": self.requested,
            "readback": self.readback,
            "warning": self.warning,
        }


def _timestamp(settings: MetadataSettings, value: Optional[datetime]) -> str:
    if settings.create_date:
        return settings.create_date
    moment = value or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y:%m:%d %H:%M:%S")


def inject_metadata(
    video_path: Path,
    settings: MetadataSettings,
    capture_time: Optional[datetime] = None,
) -> MetadataResult:
    timestamp = _timestamp(settings, capture_time)
    requested = {
        "Make": settings.make,
        "Model": settings.model,
        "Software": settings.software,
        "HandlerDescription": settings.handler_description,
        "CreateDate": timestamp,
        "ModifyDate": timestamp,
    }
    if not settings.enabled:
        return MetadataResult(
            applied=False,
            mode="disabled",
            requested=requested,
            readback={},
        )
    exiftool = shutil.which("exiftool")
    if not exiftool:
        return MetadataResult(
            applied=False,
            mode="skipped",
            requested=requested,
            readback={},
            warning="未安装 ExifTool，已跳过元数据注入。",
        )
    video_path = Path(video_path).resolve()
    try:
        run_command(
            [
                exiftool,
                "-overwrite_original",
                f"-Keys:Make={settings.make}",
                f"-Keys:Model={settings.model}",
                f"-Keys:Software={settings.software}",
                f"-QuickTime:CreateDate={timestamp}",
                f"-QuickTime:ModifyDate={timestamp}",
                f"-QuickTime:TrackCreateDate={timestamp}",
                f"-QuickTime:TrackModifyDate={timestamp}",
                f"-QuickTime:MediaCreateDate={timestamp}",
                f"-QuickTime:MediaModifyDate={timestamp}",
                str(video_path),
            ],
            "注入手机元数据",
        )
        read_result = run_command(
            [
                exiftool,
                "-j",
                "-G1",
                "-s",
                "-Make",
                "-Model",
                "-Software",
                "-CreateDate",
                "-ModifyDate",
                "-HandlerDescription",
                str(video_path),
            ],
            "回读视频元数据",
        )
        values = json.loads(read_result.stdout or "[]")
        readback = values[0] if values else {}
        readback.pop("SourceFile", None)
        return MetadataResult(
            applied=True,
            mode="exiftool",
            requested=requested,
            readback=readback,
        )
    except Exception as exc:
        return MetadataResult(
            applied=False,
            mode="skipped",
            requested=requested,
            readback={},
            warning=f"ExifTool 写入失败，已保留成片：{redact_error(exc)}",
        )


inject_phone_metadata = inject_metadata
