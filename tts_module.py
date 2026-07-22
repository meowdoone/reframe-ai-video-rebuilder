from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from config import TTSSettings
from utils import redact_error


@dataclass
class TTSResult:
    audio_path: Optional[Path]
    mode: str
    voice: str
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_file": self.audio_path.name if self.audio_path else None,
            "mode": self.mode,
            "voice": self.voice,
            "warning": self.warning,
        }


async def synthesize_speech_async(
    text: str,
    output_path: Path,
    settings: TTSSettings,
) -> TTSResult:
    narration = text.strip()
    if not narration:
        return TTSResult(
            audio_path=None,
            mode="skipped",
            voice=settings.voice,
            warning="没有可合成的旁白文本。",
        )
    try:
        import edge_tts

        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        communicator = edge_tts.Communicate(
            narration,
            settings.voice,
            rate=settings.rate,
            volume=settings.volume,
        )
        await communicator.save(str(output_path))
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("edge-tts 未生成有效音频文件")
        return TTSResult(
            audio_path=output_path,
            mode="edge-tts",
            voice=settings.voice,
        )
    except Exception as exc:
        return TTSResult(
            audio_path=None,
            mode="skipped",
            voice=settings.voice,
            warning=redact_error(exc),
        )


def synthesize_speech(
    text: str,
    output_path: Path,
    settings: TTSSettings,
) -> TTSResult:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(synthesize_speech_async(text, output_path, settings))
    raise RuntimeError(
        "当前线程已有异步事件循环，请调用 synthesize_speech_async。"
    )
