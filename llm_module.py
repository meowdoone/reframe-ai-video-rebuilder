from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import LLMSettings
from utils import redact_error


@dataclass
class RewriteResult:
    text: str
    mode: str
    model: Optional[str]
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "mode": self.mode,
            "model": self.model,
            "warning": self.warning,
        }


def public_ai_error(error: Exception) -> str:
    message = str(error)
    lowered = message.lower()
    if "401" in message or "invalid_api_key" in lowered:
        return "大模型 API 凭据无效（401）。"
    if "429" in message or "rate_limit" in lowered:
        return "大模型 API 当前限流或额度不足（429）。"
    return redact_error(error)


def build_rewrite_prompt(
    transcript: str,
    target_seconds: Optional[float] = None,
) -> str:
    duration_instruction = (
        f"将成稿控制在约 {target_seconds:.0f} 秒口播长度。"
        if target_seconds
        else "保持短促，删除重复信息。"
    )
    return f"""
请把下面的原视频转写改写成全新的短视频解说词。

硬性要求：
1. 把第一人称经历改为第三方解说/观察者视角，不保留原句式。
2. 开头前三秒直接制造悬念、反差或结果钩子。
3. 只保留可由原文支持的事实，不虚构产品效果、数据或人物经历。
4. 使用自然口语、短句和清晰节奏，适合直接交给 TTS 朗读。
5. 中段给出具体信息或视觉证明，结尾给出明确收束或行动提示。
6. {duration_instruction}
7. 只输出最终解说词，不要标题、序号、解释或 Markdown。

原始转写：
{transcript.strip()}
""".strip()


def rewrite_narration(
    transcript: str,
    settings: LLMSettings,
    target_seconds: Optional[float] = None,
) -> RewriteResult:
    source_text = transcript.strip()
    if not source_text:
        return RewriteResult(text="", mode="skipped", model=None, warning="没有可重写文本。")
    if not settings.api_key:
        return RewriteResult(
            text=source_text,
            mode="source-fallback",
            model=None,
            warning="未配置大模型 API key，已沿用 ASR 文本。",
        )
    try:
        from openai import OpenAI

        client_kwargs: Dict[str, Any] = {"api_key": settings.api_key}
        if settings.api_base:
            client_kwargs["base_url"] = settings.api_base.rstrip("/")
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=settings.model,
            messages=[
                {
                    "role": "system",
                    "content": "你是短视频解说词重构导演，负责重写叙事而不是同义替换。",
                },
                {
                    "role": "user",
                    "content": build_rewrite_prompt(source_text, target_seconds),
                },
            ],
            temperature=max(0.0, min(1.5, float(settings.temperature))),
            max_tokens=max(128, int(settings.max_tokens)),
        )
        rewritten = str(response.choices[0].message.content or "").strip()
        if not rewritten:
            raise ValueError("大模型返回了空文本")
        return RewriteResult(
            text=rewritten,
            mode="llm-rewrite",
            model=settings.model,
        )
    except Exception as exc:
        return RewriteResult(
            text=source_text,
            mode="source-fallback",
            model=settings.model,
            warning=public_ai_error(exc),
        )
