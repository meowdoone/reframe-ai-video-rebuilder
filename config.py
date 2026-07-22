from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar


@dataclass
class ASRSettings:
    backend: str = "auto"
    model: str = "base"
    language: str = "zh"
    device: str = "auto"
    compute_type: str = "int8"
    default_text: str = ""


@dataclass
class LLMSettings:
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1200


@dataclass
class TTSSettings:
    voice: str = "zh-CN-YunxiNeural"
    rate: str = "+0%"
    volume: str = "+0%"


@dataclass
class RenderSettings:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    crop_scale: float = 0.95
    contrast: float = 1.05
    saturation: float = 1.08
    brightness: float = 0.02
    crf: int = 20
    preset: str = "medium"
    handler_video: str = "Core Media Video"
    handler_audio: str = "Core Media Audio"


@dataclass
class MetadataSettings:
    enabled: bool = True
    make: str = "Apple"
    model: str = "iPhone 15 Pro"
    software: str = "17.4.1"
    handler_description: str = "Core Media Video"
    create_date: Optional[str] = None


@dataclass
class AppSettings:
    asr: ASRSettings = field(default_factory=ASRSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    tts: TTSSettings = field(default_factory=TTSSettings)
    render: RenderSettings = field(default_factory=RenderSettings)
    metadata: MetadataSettings = field(default_factory=MetadataSettings)

    def public_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["llm"]["api_key"] = "configured" if self.llm.api_key else None
        return value


SettingsType = TypeVar("SettingsType")


def _settings_from_dict(cls: Type[SettingsType], value: Any) -> SettingsType:
    if value is None:
        return cls()
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__} 配置必须是 JSON 对象。")
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{cls.__name__} 包含未知字段：{', '.join(unknown)}")
    return cls(**value)


def _environment_overrides(settings: AppSettings) -> None:
    environment = os.environ
    settings.asr.model = environment.get("VIDEO_ASR_MODEL", settings.asr.model)
    settings.asr.language = environment.get("VIDEO_ASR_LANGUAGE", settings.asr.language)
    settings.asr.default_text = environment.get(
        "VIDEO_DEFAULT_TEXT", settings.asr.default_text
    )
    settings.llm.api_key = (
        environment.get("VIDEO_LLM_API_KEY")
        or environment.get("OPENAI_API_KEY")
        or settings.llm.api_key
    )
    settings.llm.api_base = (
        environment.get("VIDEO_LLM_API_BASE")
        or environment.get("OPENAI_BASE_URL")
        or settings.llm.api_base
    )
    settings.llm.model = environment.get("VIDEO_LLM_MODEL", settings.llm.model)
    settings.tts.voice = environment.get("VIDEO_TTS_VOICE", settings.tts.voice)


def load_config(path: Optional[Path] = None) -> AppSettings:
    raw: Dict[str, Any] = {}
    if path:
        config_path = Path(path).expanduser().resolve()
        try:
            raw_value = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"配置文件不存在：{config_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"配置文件不是有效 JSON：{exc}") from exc
        if not isinstance(raw_value, dict):
            raise ValueError("配置文件根节点必须是 JSON 对象。")
        raw = raw_value
    allowed_sections = {"asr", "llm", "tts", "render", "metadata"}
    unknown_sections = sorted(set(raw) - allowed_sections)
    if unknown_sections:
        raise ValueError(f"配置包含未知模块：{', '.join(unknown_sections)}")
    settings = AppSettings(
        asr=_settings_from_dict(ASRSettings, raw.get("asr")),
        llm=_settings_from_dict(LLMSettings, raw.get("llm")),
        tts=_settings_from_dict(TTSSettings, raw.get("tts")),
        render=_settings_from_dict(RenderSettings, raw.get("render")),
        metadata=_settings_from_dict(MetadataSettings, raw.get("metadata")),
    )
    _environment_overrides(settings)
    return settings
