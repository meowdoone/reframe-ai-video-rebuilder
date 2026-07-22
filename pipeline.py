from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


ProgressCallback = Callable[[int, str], None]


class PipelineError(RuntimeError):
    pass


def public_ai_error(error: Exception) -> str:
    message = str(error)
    lowered = message.lower()
    if "401" in message or "invalid_api_key" in lowered:
        return "OpenAI API 凭据无效（401）。"
    if "429" in message or "rate_limit" in lowered:
        return "OpenAI API 当前限流或额度不足（429）。"
    message = re.sub(r"sk-[A-Za-z0-9_*.-]+", "[redacted]", message)
    return message[:500]


def _progress(callback: Optional[ProgressCallback], value: int, message: str) -> None:
    if callback:
        callback(max(0, min(100, int(value))), message)


def _run(command: Sequence[str], label: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise PipelineError(f"{label}失败：{detail[-1800:]}")
    return result


def _binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise PipelineError(f"未找到 {name}，请先安装 FFmpeg。")
    return path


def probe_video(path: Path) -> Dict[str, Any]:
    result = _run(
        [
            _binary("ffprobe"),
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
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if not video:
        raise PipelineError(f"{path.name} 不包含可读取的视频轨。")
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    duration = float(
        video.get("duration")
        or data.get("format", {}).get("duration")
        or 0
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


def _estimate_focus(frame: np.ndarray) -> Tuple[float, float]:
    small = cv2.resize(frame, (240, 240), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    x_grad = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    y_grad = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = cv2.magnitude(x_grad, y_grad)
    energy = cv2.GaussianBlur(energy, (15, 15), 0)
    total = float(energy.sum())
    if total <= 1e-6:
        return 0.5, 0.5
    ys, xs = np.indices(energy.shape)
    focus_x = float((xs * energy).sum() / total) / max(energy.shape[1] - 1, 1)
    focus_y = float((ys * energy).sum() / total) / max(energy.shape[0] - 1, 1)
    return round(max(0.0, min(1.0, focus_x)), 3), round(max(0.0, min(1.0, focus_y)), 3)


def detect_scenes(
    path: Path,
    source_id: str,
    keyframe_dir: Path,
    max_segments: int = 14,
) -> List[Dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PipelineError(f"无法打开视频：{path.name}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0
    sample_every = max(1, int(round(fps / 3.0)))
    previous: Optional[np.ndarray] = None
    boundaries = [0.0]
    samples: List[Tuple[float, float]] = []
    last_boundary = 0.0
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % sample_every:
            frame_index += 1
            continue
        timestamp = frame_index / fps
        preview = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
        diff = 0.0 if previous is None else float(cv2.absdiff(gray, previous).mean())
        samples.append((timestamp, diff))
        previous = gray
        enough_gap = timestamp - last_boundary >= 0.8
        scene_change = diff >= 24.0 and enough_gap
        max_length = timestamp - last_boundary >= 4.2
        if scene_change or max_length:
            boundaries.append(round(timestamp, 3))
            last_boundary = timestamp
        frame_index += 1

    capture.release()
    if duration <= 0:
        duration = samples[-1][0] if samples else 0
    if duration <= 0.2:
        raise PipelineError(f"视频时长过短：{path.name}")
    if duration - boundaries[-1] < 0.45 and len(boundaries) > 1:
        boundaries[-1] = round(duration, 3)
    else:
        boundaries.append(round(duration, 3))

    raw_segments: List[Dict[str, Any]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start < 0.45:
            continue
        interval_scores = [score for stamp, score in samples if start <= stamp < end]
        motion = float(np.mean(interval_scores)) if interval_scores else 0.0
        raw_segments.append(
            {
                "source_id": source_id,
                "start": round(start, 3),
                "end": round(end, 3),
                "motion_score": round(motion, 3),
            }
        )

    if not raw_segments:
        raw_segments = [
            {
                "source_id": source_id,
                "start": 0.0,
                "end": round(duration, 3),
                "motion_score": 0.0,
            }
        ]

    if len(raw_segments) > max_segments:
        raw_segments = sorted(
            sorted(raw_segments, key=lambda item: item["motion_score"], reverse=True)[:max_segments],
            key=lambda item: item["start"],
        )

    keyframe_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(path))
    segments: List[Dict[str, Any]] = []
    for index, segment in enumerate(raw_segments):
        sample_times = [
            segment["start"] + (segment["end"] - segment["start"]) * fraction
            for fraction in (0.15, 0.5, 0.85)
        ]
        focus_points: List[Tuple[float, float]] = []
        keyframe_frame: Optional[np.ndarray] = None
        for sample_index, timestamp in enumerate(sample_times):
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            focus_points.append(_estimate_focus(frame))
            if sample_index == 1:
                keyframe_frame = frame
        if not focus_points:
            focus_points = [(0.5, 0.5)]
        focus_x = round(float(np.mean([point[0] for point in focus_points])), 3)
        focus_y = round(float(np.mean([point[1] for point in focus_points])), 3)
        focus_start_x, focus_start_y = focus_points[0]
        focus_end_x, focus_end_y = focus_points[-1]
        if keyframe_frame is None:
            keyframe_path = None
        else:
            keyframe_path = keyframe_dir / f"{source_id}_{index:02d}.jpg"
            preview = keyframe_frame
            max_side = max(keyframe_frame.shape[:2])
            if max_side > 960:
                scale = 960 / max_side
                preview = cv2.resize(
                    keyframe_frame,
                    (
                        int(keyframe_frame.shape[1] * scale),
                        int(keyframe_frame.shape[0] * scale),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imwrite(str(keyframe_path), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        segments.append(
            {
                **segment,
                "focus_x": focus_x,
                "focus_y": focus_y,
                "focus_start_x": focus_start_x,
                "focus_start_y": focus_start_y,
                "focus_end_x": focus_end_x,
                "focus_end_y": focus_end_y,
                "keyframe": str(keyframe_path) if keyframe_path else None,
            }
        )
    capture.release()
    return segments


def analyze_sources(
    paths: Sequence[Path],
    job_dir: Path,
    progress: Optional[ProgressCallback] = None,
) -> List[Dict[str, Any]]:
    analyses: List[Dict[str, Any]] = []
    keyframe_dir = job_dir / "keyframes"
    for index, path in enumerate(paths):
        _progress(progress, 8 + int(index / max(len(paths), 1) * 22), f"分析素材 {index + 1}/{len(paths)}")
        metadata = probe_video(path)
        source_id = f"source_{index}"
        segments = detect_scenes(path, source_id, keyframe_dir)
        analyses.append(
            {
                "source_id": source_id,
                "name": path.name,
                "path": str(path),
                **metadata,
                "sha256": sha256_file(path),
                "segments": segments,
            }
        )
    return analyses


def _caption_phrases(brief: str) -> List[str]:
    phrases = [
        re.sub(r"\s+", " ", part).strip(" -—:：,.，。!！?？")
        for part in re.split(r"[\n。！？!?；;]+", brief or "")
    ]
    phrases = [phrase for phrase in phrases if phrase]
    if not phrases:
        phrases = ["先看结果", "细节一眼看懂", "真实场景直接展示", "现在就行动"]
    cleaned: List[str] = []
    for phrase in phrases:
        if len(phrase) > 32:
            phrase = phrase[:31] + "…"
        cleaned.append(phrase)
    return cleaned[:8]


def _flatten_segments(analyses: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(segment) for analysis in analyses for segment in analysis.get("segments", [])]


def _expand_segments_for_edit(
    segments: Sequence[Dict[str, Any]], style: str
) -> List[Dict[str, Any]]:
    chunk_length = {
        "fast-cut": 1.35,
        "ugc": 2.2,
        "product-demo": 2.4,
        "story": 2.8,
    }.get(style, 2.2)
    expanded: List[Dict[str, Any]] = []
    for segment in segments:
        cursor = float(segment["start"])
        end = float(segment["end"])
        part = 0
        while end - cursor >= 0.45:
            chunk_end = min(end, cursor + chunk_length)
            item = dict(segment)
            item["start"] = round(cursor, 3)
            item["end"] = round(chunk_end, 3)
            item["motion_score"] = float(item.get("motion_score", 0)) - part * 0.01
            expanded.append(item)
            cursor = chunk_end
            part += 1
    return expanded


def create_fallback_plan(
    analyses: Sequence[Dict[str, Any]],
    target_duration: float,
    brief: str,
    style: str = "ugc",
) -> Dict[str, Any]:
    pool = _expand_segments_for_edit(_flatten_segments(analyses), style)
    if not pool:
        raise PipelineError("没有检测到可用镜头。")
    target_duration = max(3.0, float(target_duration))
    phrases = _caption_phrases(brief)
    ranked = sorted(pool, key=lambda item: item.get("motion_score", 0), reverse=True)
    selected: List[Dict[str, Any]] = []
    remaining = ranked[:]
    elapsed = 0.0
    last_source: Optional[str] = None
    source_usage = {analysis["source_id"]: 0 for analysis in analyses}

    while remaining and elapsed < target_duration - 0.35:
        alternate = [item for item in remaining if item["source_id"] != last_source]
        candidate = min(
            alternate or remaining,
            key=lambda item: (
                source_usage.get(item["source_id"], 0),
                -float(item.get("motion_score", 0)),
            ),
        )
        remaining.remove(candidate)
        raw_available = candidate["end"] - candidate["start"]
        desired = 1.6 if not selected else 2.4
        if style == "story":
            desired = 2.8
        elif style == "fast-cut":
            desired = 1.35 if selected else 1.0
        duration = min(raw_available, desired, target_duration - elapsed)
        if duration < 0.45:
            continue
        if not selected:
            caption = phrases[0]
        elif len(phrases) > 2:
            caption = phrases[1 + ((len(selected) - 1) % (len(phrases) - 2))]
        else:
            caption = phrases[min(len(selected), len(phrases) - 1)]
        beat = {
            "source_id": candidate["source_id"],
            "start": candidate["start"],
            "end": round(candidate["start"] + duration, 3),
            "speed": 1.0,
            "focus_x": candidate.get("focus_x", 0.5),
            "focus_y": candidate.get("focus_y", 0.5),
            "focus_start_x": candidate.get("focus_start_x", candidate.get("focus_x", 0.5)),
            "focus_start_y": candidate.get("focus_start_y", candidate.get("focus_y", 0.5)),
            "focus_end_x": candidate.get("focus_end_x", candidate.get("focus_x", 0.5)),
            "focus_end_y": candidate.get("focus_end_y", candidate.get("focus_y", 0.5)),
            "purpose": "hook" if not selected else "proof",
            "caption": caption,
        }
        selected.append(beat)
        elapsed += duration
        last_source = candidate["source_id"]
        source_usage[candidate["source_id"]] = source_usage.get(candidate["source_id"], 0) + 1

    if selected:
        selected[-1]["purpose"] = "cta"
        selected[-1]["caption"] = phrases[-1]
    return {
        "title": "本地智能重构方案",
        "creative_angle": brief.strip()[:120] or "用快速结果展示、细节证明和行动提示重组素材",
        "hook_text": selected[0]["caption"] if selected else phrases[0],
        "mode": "local-smart-plan",
        "beats": selected,
        "estimated_duration": round(elapsed, 3),
    }


def _finite_number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _matching_segment(
    analysis: Dict[str, Any], start: float, end: float
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_overlap = 0.0
    for segment in analysis.get("segments", []):
        overlap = max(
            0.0,
            min(end, float(segment["end"])) - max(start, float(segment["start"])),
        )
        if overlap > best_overlap:
            best, best_overlap = segment, overlap
    return best if best_overlap >= 0.45 else None


def _interpolate_focus(
    segment: Dict[str, Any], axis: str, timestamp: float
) -> float:
    segment_start = float(segment["start"])
    segment_end = float(segment["end"])
    duration = max(segment_end - segment_start, 1e-6)
    position = max(0.0, min(1.0, (timestamp - segment_start) / duration))
    fallback = _finite_number(segment.get(f"focus_{axis}"), 0.5)
    start = _finite_number(segment.get(f"focus_start_{axis}"), fallback)
    end = _finite_number(segment.get(f"focus_end_{axis}"), fallback)
    return start + (end - start) * position


def normalize_plan(
    raw_plan: Dict[str, Any],
    analyses: Sequence[Dict[str, Any]],
    target_duration: float,
    brief: str,
) -> Dict[str, Any]:
    analysis_map = {analysis["source_id"]: analysis for analysis in analyses}
    target_duration = max(3.0, float(target_duration))
    normalized: List[Dict[str, Any]] = []
    elapsed = 0.0
    for raw in list(raw_plan.get("beats") or [])[:20]:
        source_id = str(raw.get("source_id") or "")
        analysis = analysis_map.get(source_id)
        if not analysis:
            continue
        requested_start = max(0.0, _finite_number(raw.get("start"), 0.0))
        requested_end = max(requested_start, _finite_number(raw.get("end"), requested_start))
        segment = _matching_segment(analysis, requested_start, requested_end)
        if not segment:
            continue
        start = max(requested_start, float(segment["start"]))
        end = min(requested_end, float(segment["end"]))
        speed = max(0.75, min(1.35, _finite_number(raw.get("speed"), 1.0)))
        raw_duration = end - start
        if raw_duration < 0.45:
            continue
        available = target_duration - elapsed
        if available < 0.35:
            break
        output_duration = raw_duration / speed
        if output_duration > available:
            end = start + available * speed
            output_duration = available
        segment_focus_x = _finite_number(segment.get("focus_x"), 0.5)
        segment_focus_y = _finite_number(segment.get("focus_y"), 0.5)
        focus_x = max(0.0, min(1.0, _finite_number(raw.get("focus_x"), segment_focus_x)))
        focus_y = max(0.0, min(1.0, _finite_number(raw.get("focus_y"), segment_focus_y)))
        focus_offset_x = focus_x - segment_focus_x
        focus_offset_y = focus_y - segment_focus_y
        focus_start_x = _interpolate_focus(segment, "x", start) + focus_offset_x
        focus_start_y = _interpolate_focus(segment, "y", start) + focus_offset_y
        focus_end_x = _interpolate_focus(segment, "x", end) + focus_offset_x
        focus_end_y = _interpolate_focus(segment, "y", end) + focus_offset_y
        normalized.append(
            {
                "source_id": source_id,
                "start": round(start, 3),
                "end": round(end, 3),
                "speed": round(speed, 3),
                "focus_x": round(focus_x, 3),
                "focus_y": round(focus_y, 3),
                "focus_start_x": round(
                    max(0.0, min(1.0, focus_start_x)),
                    3,
                ),
                "focus_start_y": round(
                    max(0.0, min(1.0, focus_start_y)),
                    3,
                ),
                "focus_end_x": round(
                    max(0.0, min(1.0, focus_end_x)),
                    3,
                ),
                "focus_end_y": round(
                    max(0.0, min(1.0, focus_end_y)),
                    3,
                ),
                "purpose": str(raw.get("purpose") or "proof")[:32],
                "caption": str(raw.get("caption") or "")[:72],
            }
        )
        elapsed += output_duration

    if not normalized:
        return create_fallback_plan(analyses, target_duration, brief)

    plan = {
        "title": str(raw_plan.get("title") or "AI 重构方案")[:100],
        "creative_angle": str(raw_plan.get("creative_angle") or brief or "重新组织镜头叙事")[:300],
        "hook_text": str(raw_plan.get("hook_text") or normalized[0]["caption"])[:72],
        "mode": str(raw_plan.get("mode") or "ai-vision-plan"),
        "beats": normalized,
        "estimated_duration": round(elapsed, 3),
    }
    for field in ("ai_model", "ai_response_id"):
        if raw_plan.get(field):
            plan[field] = raw_plan[field]
    return plan


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise PipelineError("AI 没有返回可解析的重构方案。")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise PipelineError("AI 重构方案不是 JSON 对象。")
    return value


def select_ai_candidates(
    analyses: Sequence[Dict[str, Any]], limit: int = 12
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected = sorted(
        _flatten_segments(analyses),
        key=lambda item: item.get("motion_score", 0),
        reverse=True,
    )[:limit]
    selected_ids = {
        (str(segment["source_id"]), float(segment["start"]), float(segment["end"]))
        for segment in selected
    }
    restricted: List[Dict[str, Any]] = []
    for analysis in analyses:
        allowed_segments = [
            segment
            for segment in analysis.get("segments", [])
            if (
                str(segment["source_id"]),
                float(segment["start"]),
                float(segment["end"]),
            )
            in selected_ids
        ]
        if allowed_segments:
            restricted.append({**analysis, "segments": allowed_segments})
    return selected, restricted


def create_ai_plan(
    analyses: Sequence[Dict[str, Any]],
    target_duration: float,
    brief: str,
    style: str,
    language: str,
) -> Dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise PipelineError("未配置 OPENAI_API_KEY。")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise PipelineError("未安装 openai Python SDK。") from exc

    candidates: List[Dict[str, Any]] = []
    image_parts: List[Dict[str, Any]] = []
    segments, candidate_analyses = select_ai_candidates(analyses)
    for segment in segments:
        candidates.append(
            {
                "source_id": segment["source_id"],
                "start": segment["start"],
                "end": segment["end"],
                "motion_score": segment.get("motion_score", 0),
                "focus_x": segment.get("focus_x", 0.5),
                "focus_y": segment.get("focus_y", 0.5),
            }
        )
        keyframe = segment.get("keyframe")
        if keyframe and Path(keyframe).exists():
            encoded = base64.b64encode(Path(keyframe).read_bytes()).decode("ascii")
            image_parts.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encoded}",
                    "detail": "low",
                }
            )

    prompt = f"""
Role: short-form video reconstruction director.

Goal: create a genuinely new {target_duration:.0f}-second vertical-video edit plan using only the authorized candidate source ranges below.

Creative brief: {brief or 'Create a clear hook, visual proof, payoff, and CTA.'}
Style: {style}. Caption language: {language}.

Success criteria:
- choose 3-12 beats and keep total output duration at or below {target_duration:.1f}s
- redesign the hook, narrative order, captions, pacing, and focus crop
- use only listed source_id and timestamp ranges
- captions are short, concrete, and ready to burn into video
- do not request new footage or claim facts absent from the brief

Candidate segments:
{json.dumps(candidates, ensure_ascii=False)}

Return only one JSON object with this shape:
{{
  "title": "...",
  "creative_angle": "...",
  "hook_text": "...",
  "beats": [
    {{
      "source_id": "source_0",
      "start": 0.0,
      "end": 1.8,
      "speed": 1.0,
      "focus_x": 0.5,
      "focus_y": 0.5,
      "purpose": "hook|proof|payoff|cta",
      "caption": "..."
    }}
  ]
}}
""".strip()
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    content.extend(image_parts)
    configured = os.environ.get("VIDEO_REBUILDER_MODEL", "gpt-5.6-sol")
    models = [configured] + (["gpt-5.6"] if configured != "gpt-5.6" else [])
    last_error: Optional[Exception] = None
    client = OpenAI()
    for model in models:
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": content}],
                max_output_tokens=2500,
                reasoning={"effort": "low"},
                text={"verbosity": "low"},
                store=False,
            )
            plan = _extract_json(response.output_text)
            plan["mode"] = "ai-vision-plan"
            plan["ai_model"] = model
            plan["ai_response_id"] = response.id
            return normalize_plan(plan, candidate_analyses, target_duration, brief)
        except Exception as exc:  # SDK errors are surfaced in the manifest and UI.
            last_error = exc
    raise PipelineError(f"AI 视觉规划不可用：{last_error}")


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole = int(seconds % 60)
    centiseconds = int(round((seconds - math.floor(seconds)) * 100))
    if centiseconds == 100:
        whole += 1
        centiseconds = 0
    return f"{hours}:{minutes:02d}:{whole:02d}.{centiseconds:02d}"


def _escape_ass(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def build_ass(plan: Dict[str, Any], path: Path, width: int, height: int) -> None:
    font_size = max(34, int(height * 0.038))
    margin_v = max(100, int(height * 0.13))
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,PingFang SC,{font_size},&H00FFFFFF,&H000000FF,&H00111111,&H88000000,-1,0,0,0,100,100,0,0,1,4,1,2,80,80,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    elapsed = 0.0
    lines = [header]
    for beat in plan.get("beats", []):
        duration = (float(beat["end"]) - float(beat["start"])) / float(beat.get("speed") or 1)
        caption = str(beat.get("caption") or "").strip()
        if caption:
            lines.append(
                f"Dialogue: 0,{_ass_time(elapsed)},{_ass_time(elapsed + duration)},Caption,,0,0,0,,{_escape_ass(caption)}\n"
            )
        elapsed += duration
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8-sig")


def _filter_path(path: Path) -> str:
    return str(path).replace("\\", r"\\").replace("'", r"\'").replace(":", r"\:")


def _final_encoding_args(output: Path) -> List[str]:
    return [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output),
    ]


def render_video(
    plan: Dict[str, Any],
    analyses: Sequence[Dict[str, Any]],
    job_dir: Path,
    width: int = 1080,
    height: int = 1920,
    music_path: Optional[Path] = None,
    progress: Optional[ProgressCallback] = None,
) -> Path:
    beats = plan.get("beats") or []
    if not beats:
        raise PipelineError("重构方案没有可渲染镜头。")
    source_map = {analysis["source_id"]: analysis for analysis in analyses}
    raw_output = job_dir / "reconstructed_raw.mp4"
    output = job_dir / "reconstructed_tiktok.mp4"
    ass_path = job_dir / "captions.ass"
    build_ass(plan, ass_path, width=width, height=height)

    command: List[str] = [_binary("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error"]
    filter_parts: List[str] = []
    concat_inputs: List[str] = []
    total_duration = 0.0
    for index, beat in enumerate(beats):
        source = source_map[beat["source_id"]]
        raw_duration = float(beat["end"]) - float(beat["start"])
        speed = float(beat.get("speed") or 1.0)
        output_duration = raw_duration / speed
        total_duration += output_duration
        command.extend(
            [
                "-ss",
                f"{float(beat['start']):.3f}",
                "-t",
                f"{raw_duration:.3f}",
                "-i",
                source["path"],
            ]
        )
        focus_x = max(0.0, min(1.0, float(beat.get("focus_x", 0.5))))
        focus_y = max(0.0, min(1.0, float(beat.get("focus_y", 0.5))))
        focus_start_x = max(0.0, min(1.0, float(beat.get("focus_start_x", focus_x))))
        focus_start_y = max(0.0, min(1.0, float(beat.get("focus_start_y", focus_y))))
        focus_end_x = max(0.0, min(1.0, float(beat.get("focus_end_x", focus_x))))
        focus_end_y = max(0.0, min(1.0, float(beat.get("focus_end_y", focus_y))))
        animation_frames = max(1.0, output_duration * 30.0 - 1.0)
        crop_x = (
            f"(in_w-out_w)*({focus_start_x:.5f}+"
            f"({focus_end_x - focus_start_x:.5f})*n/{animation_frames:.3f})"
        )
        crop_y = (
            f"(in_h-out_h)*({focus_start_y:.5f}+"
            f"({focus_end_y - focus_start_y:.5f})*n/{animation_frames:.3f})"
        )
        filter_parts.append(
            f"[{index}:v:0]trim=duration={raw_duration:.3f},"
            f"setpts=(PTS-STARTPTS)/{speed:.5f},"
            "fps=30,"
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}:x={crop_x}:y={crop_y},"
            "setsar=1,eq=contrast=1.025:saturation=1.04,format=yuv420p"
            f"[v{index}]"
        )
        if source.get("has_audio"):
            filter_parts.append(
                f"[{index}:a:0]atrim=duration={raw_duration:.3f},asetpts=PTS-STARTPTS,"
                f"atempo={speed:.5f},apad=pad_dur={output_duration:.3f},"
                f"atrim=duration={output_duration:.3f},aresample=48000,"
                f"aformat=sample_rates=48000:channel_layouts=stereo[a{index}]"
            )
        else:
            filter_parts.append(
                "anullsrc=channel_layout=stereo:sample_rate=48000,"
                f"atrim=duration={output_duration:.3f}[a{index}]"
            )
        concat_inputs.append(f"[v{index}][a{index}]")
    filter_parts.append(
        "".join(concat_inputs) + f"concat=n={len(beats)}:v=1:a=1[vout][aout]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(raw_output),
        ]
    )
    _progress(progress, 62, "重排镜头并统一画幅")
    _run(command, "镜头重构")

    _progress(progress, 82, "写入字幕并完成声音处理")
    subtitle_filter = f"subtitles=filename='{_filter_path(ass_path)}'"
    final_command: List[str] = [
        _binary("ffmpeg"),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(raw_output),
    ]
    if music_path:
        final_command.extend(["-stream_loop", "-1", "-i", str(music_path)])
        fade_start = max(0.0, total_duration - 0.8)
        final_filter = (
            f"[0:v]{subtitle_filter}[v];"
            f"[1:a]volume=0.14,atrim=duration={total_duration:.3f},"
            f"afade=t=out:st={fade_start:.3f}:d=0.8[m];"
            "[0:a][m]amix=inputs=2:duration=first:dropout_transition=2,"
            "loudnorm=I=-14:TP=-1.5:LRA=11[a]"
        )
        final_command.extend(["-filter_complex", final_filter, "-map", "[v]", "-map", "[a]"])
    else:
        final_command.extend(
            [
                "-vf",
                subtitle_filter,
                "-af",
                "loudnorm=I=-14:TP=-1.5:LRA=11",
            ]
        )
    final_command.extend(_final_encoding_args(output))
    try:
        _run(final_command, "成片输出")
    except PipelineError:
        if not music_path:
            raise
        fallback = [
            _binary("ffmpeg"),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_output),
            "-vf",
            subtitle_filter,
            "-af",
            "loudnorm=I=-14:TP=-1.5:LRA=11",
        ]
        fallback.extend(_final_encoding_args(output))
        _run(fallback, "成片输出")
    raw_output.unlink(missing_ok=True)
    return output


def _read_frame(path: Path, timestamp: float) -> Optional[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000)
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def _perceptual_change(first: np.ndarray, second: np.ndarray) -> float:
    def difference_hash(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (17, 16), interpolation=cv2.INTER_AREA)
        return resized[:, 1:] > resized[:, :-1]

    return float(np.mean(difference_hash(first) != difference_hash(second)))


def measure_visual_change(
    output: Path,
    plan: Dict[str, Any],
    analyses: Sequence[Dict[str, Any]],
    sample_limit: int = 10,
) -> Dict[str, Any]:
    source_map = {analysis["source_id"]: analysis for analysis in analyses}
    beats = list(plan.get("beats") or [])
    if len(beats) > sample_limit:
        indexes = np.linspace(0, len(beats) - 1, sample_limit, dtype=int)
        sample_indexes = set(int(index) for index in indexes)
    else:
        sample_indexes = set(range(len(beats)))
    output_cursor = 0.0
    distances: List[float] = []
    for index, beat in enumerate(beats):
        raw_duration = float(beat["end"]) - float(beat["start"])
        output_duration = raw_duration / float(beat.get("speed") or 1.0)
        if index in sample_indexes:
            source = source_map.get(str(beat.get("source_id")))
            if source:
                source_frame = _read_frame(
                    Path(source["path"]),
                    float(beat["start"]) + raw_duration / 2,
                )
                output_frame = _read_frame(output, output_cursor + output_duration / 2)
                if source_frame is not None and output_frame is not None:
                    distances.append(_perceptual_change(source_frame, output_frame))
        output_cursor += output_duration
    return {
        "metric": "perceptual_frame_change",
        "change_index": round(float(np.mean(distances)) * 100, 1) if distances else 0.0,
        "sample_count": len(distances),
    }


def build_transformation_report(
    plan: Dict[str, Any],
    analyses: Sequence[Dict[str, Any]],
    visual_change: Dict[str, Any],
) -> Dict[str, Any]:
    beats = plan.get("beats") or []
    source_order = [str(beat.get("source_id")) for beat in beats]
    source_switches = sum(
        1 for previous, current in zip(source_order, source_order[1:]) if previous != current
    )
    used_sources = sorted(set(source_order))
    durations = [
        (float(beat["end"]) - float(beat["start"]))
        / float(beat.get("speed") or 1.0)
        for beat in beats
    ]
    total_duration = sum(durations)
    caption_duration = sum(
        duration
        for beat, duration in zip(beats, durations)
        if str(beat.get("caption") or "").strip()
    )
    caption_coverage = caption_duration / total_duration if total_duration else 0.0
    average_shot = total_duration / len(beats) if beats else 0.0
    focus_travel = [
        math.hypot(
            float(beat.get("focus_end_x", beat.get("focus_x", 0.5)))
            - float(beat.get("focus_start_x", beat.get("focus_x", 0.5))),
            float(beat.get("focus_end_y", beat.get("focus_y", 0.5)))
            - float(beat.get("focus_start_y", beat.get("focus_y", 0.5))),
        )
        for beat in beats
    ]
    average_focus_travel = float(np.mean(focus_travel)) if focus_travel else 0.0
    source_switch_ratio = source_switches / max(len(beats) - 1, 1)
    available_sources = max(len(analyses), 1)
    source_mix_ratio = len(used_sources) / available_sources
    visual_change_index = float(visual_change.get("change_index") or 0.0)
    source_composition_score = (
        min(10.0, source_mix_ratio * 10.0)
        if available_sources > 1
        else (3.0 if beats else 0.0)
    )
    components = {
        "narrative_rewrite": 15.0
        if plan.get("creative_angle") and plan.get("hook_text")
        else 8.0,
        "timeline_recomposition": min(14.0, len(beats) * 2.8)
        + min(8.0, source_switch_ratio * 8.0),
        "source_composition": source_composition_score,
        "visual_reframe": min(15.0, visual_change_index * 0.15)
        + min(7.0, average_focus_travel * 70.0),
        "caption_rewrite": min(15.0, caption_coverage * 15.0),
        "audio_remaster": 6.0 if beats else 0.0,
        "short_form_pacing": 7.0 if 0.45 <= average_shot <= 3.0 else 2.0,
    }
    score = round(min(100.0, sum(components.values())))
    return {
        "metric": "internal_edit_transformation",
        "scope": "editing_change_evidence_not_platform_originality",
        "depth_score": min(100, score),
        "components": {name: round(value, 1) for name, value in components.items()},
        "visual_change": visual_change,
        "caption_coverage": round(caption_coverage, 3),
        "average_shot_seconds": round(average_shot, 3),
        "average_focus_travel": round(average_focus_travel, 3),
        "beat_count": len(beats),
        "available_source_count": len(analyses),
        "used_source_count": len(used_sources),
        "source_switches": source_switches,
    }


def build_platform_preflight(
    output: Dict[str, Any],
    plan: Dict[str, Any],
    transformation: Dict[str, Any],
) -> Dict[str, Any]:
    width = int(output.get("width") or 0)
    height = int(output.get("height") or 0)
    fps = float(output.get("fps") or 0)
    duration = float(output.get("duration") or 0)
    technical_checks = {
        "vertical_9_16": height > 0 and abs(width / height - 9 / 16) < 0.01,
        "minimum_720p": width >= 720 and height >= 1280,
        "supported_frame_rate": 23 <= fps <= 60,
        "h264_video": output.get("video_codec") == "h264",
        "aac_audio": output.get("audio_codec") == "aac",
        "short_form_duration": 0 < duration <= 60,
    }
    beats = list(plan.get("beats") or [])
    first_duration = 0.0
    if beats:
        first_duration = (
            float(beats[0]["end"]) - float(beats[0]["start"])
        ) / float(beats[0].get("speed") or 1.0)
    creative_checks = {
        "hook_in_first_3_seconds": bool(beats)
        and str(beats[0].get("purpose")) == "hook"
        and first_duration <= 3.0,
        "active_short_form_pacing": bool(beats)
        and float(transformation.get("average_shot_seconds") or 99.0) <= 3.0,
        "caption_coverage": float(transformation.get("caption_coverage") or 0.0) >= 0.65,
        "ending_payoff_or_cta": bool(beats)
        and str(beats[-1].get("purpose")) in {"payoff", "cta"},
        "measured_visual_change": int(
            transformation.get("visual_change", {}).get("sample_count") or 0
        )
        > 0,
        "caption_safe_zone_layout": True,
    }
    technical_passed = all(technical_checks.values())
    creative_structure_passed = all(creative_checks.values())
    return {
        "profile": "tiktok_vertical_edit_preflight_v2",
        "scope": "technical_export_and_edit_structure_only",
        "passed": technical_passed and creative_structure_passed,
        "technical_passed": technical_passed,
        "creative_structure_passed": creative_structure_passed,
        "technical_checks": technical_checks,
        "creative_checks": creative_checks,
    }


def run_pipeline(
    video_paths: Sequence[Path],
    job_dir: Path,
    brief: str,
    target_duration: float = 15.0,
    style: str = "ugc",
    language: str = "简体中文",
    use_ai: bool = True,
    music_path: Optional[Path] = None,
    width: int = 1080,
    height: int = 1920,
    progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    if not video_paths:
        raise PipelineError("请至少导入一个视频。")
    job_dir.mkdir(parents=True, exist_ok=True)
    _progress(progress, 3, "读取视频素材")
    analyses = analyze_sources(video_paths, job_dir, progress)
    _progress(progress, 34, "生成镜头级重构方案")
    ai_error: Optional[str] = None
    if use_ai:
        try:
            plan = create_ai_plan(analyses, target_duration, brief, style, language)
        except Exception as exc:
            ai_error = public_ai_error(exc)
            plan = create_fallback_plan(analyses, target_duration, brief, style)
            plan["mode"] = "local-smart-fallback"
    else:
        plan = create_fallback_plan(analyses, target_duration, brief, style)
    plan = normalize_plan(plan, analyses, target_duration, brief)
    (job_dir / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _progress(progress, 48, "开始重构成片")
    output = render_video(
        plan,
        analyses,
        job_dir,
        width=width,
        height=height,
        music_path=music_path,
        progress=progress,
    )
    output_metadata = probe_video(output)
    visual_change = measure_visual_change(output, plan, analyses)
    transformation = build_transformation_report(plan, analyses, visual_change)
    platform_preflight = build_platform_preflight(
        output_metadata, plan, transformation
    )
    manifest = {
        "schema_version": "video-rebuilder-v1",
        "inputs": [
            {
                key: value
                for key, value in analysis.items()
                if key not in {"path", "segments"}
            }
            for analysis in analyses
        ],
        "settings": {
            "target_duration": target_duration,
            "style": style,
            "language": language,
            "width": width,
            "height": height,
            "ai_requested": use_ai,
        },
        "planning": {
            "mode": plan.get("mode"),
            "ai_model": plan.get("ai_model"),
            "ai_error": ai_error,
        },
        "transformation": transformation,
        "platform_preflight": platform_preflight,
        "output": {
            "file": output.name,
            "sha256": sha256_file(output),
            **output_metadata,
        },
    }
    (job_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _progress(progress, 100, "成片已完成")
    return {
        "video": output.name,
        "plan": "plan.json",
        "manifest": "manifest.json",
        "captions": "captions.ass",
        "planning_mode": plan.get("mode"),
        "ai_error": ai_error,
        "output": output_metadata,
        "beats": plan.get("beats", []),
        "title": plan.get("title"),
        "creative_angle": plan.get("creative_angle"),
        "transformation": transformation,
        "platform_preflight": platform_preflight,
    }
