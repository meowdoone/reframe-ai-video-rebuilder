from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from asr_module import ASRResult, transcribe_video
from config import AppSettings, load_config
from llm_module import RewriteResult, rewrite_narration
from metadata_injector import inject_metadata
from tts_module import TTSResult, synthesize_speech
from utils import (
    PipelineError,
    ProgressCallback,
    cleanup_paths,
    discover_video_files,
    emit_progress,
    probe_video,
    safe_filename,
    setup_logging,
    sha256_file,
    write_json,
)
from video_renderer import render_video


def _artifact_prefix(video_path: Path) -> str:
    stem = safe_filename(video_path.stem, "video")
    return f"{stem}_{sha256_file(video_path)[:8]}"


def process_video(
    video_path: Path,
    output_dir: Path,
    settings: AppSettings,
    *,
    skip_asr: bool = False,
    skip_llm: bool = False,
    skip_tts: bool = False,
    keep_temp: bool = False,
    overwrite: bool = False,
    progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    video_path = Path(video_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not video_path.exists():
        raise PipelineError(f"输入视频不存在：{video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _artifact_prefix(video_path)
    output_path = output_dir / f"{prefix}_reconstructed.mp4"
    manifest_path = output_dir / f"{prefix}_manifest.json"
    transcript_path = output_dir / f"{prefix}_transcript.json"
    script_path = output_dir / f"{prefix}_narration.txt"
    if output_path.exists() and not overwrite:
        raise PipelineError(f"输出已存在，请使用 --overwrite：{output_path}")

    work_root = output_dir / ".reframe-work"
    work_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{prefix}-", dir=str(work_root)))
    input_metadata = probe_video(video_path)
    created_paths = [output_path, manifest_path, transcript_path, script_path]
    try:
        emit_progress(progress, 5, "提取音频并执行 ASR")
        if skip_asr:
            text = settings.asr.default_text.strip()
            asr_result = ASRResult(
                text=text,
                segments=[],
                mode="manual-text" if text else "skipped",
                language=settings.asr.language or None,
                audio_path=None,
                warning=None if text else "已跳过 ASR 且未提供默认文本。",
            )
        else:
            asr_result = transcribe_video(video_path, work_dir, settings.asr)
        write_json(transcript_path, asr_result.to_dict())

        emit_progress(progress, 35, "重构短视频解说词")
        if skip_llm:
            rewrite_result = RewriteResult(
                text=asr_result.text,
                mode="disabled",
                model=None,
            )
        else:
            rewrite_result = rewrite_narration(
                asr_result.text,
                settings.llm,
                target_seconds=input_metadata["duration"],
            )
        script_path.write_text(rewrite_result.text, encoding="utf-8")

        emit_progress(progress, 55, "生成全新旁白音轨")
        if skip_tts:
            tts_result = TTSResult(
                audio_path=None,
                mode="disabled",
                voice=settings.tts.voice,
            )
        else:
            tts_result = synthesize_speech(
                rewrite_result.text,
                work_dir / "narration.mp3",
                settings.tts,
            )

        emit_progress(progress, 70, "重构画面并替换原音")
        render_result = render_video(
            video_path,
            tts_result.audio_path,
            output_path,
            settings.render,
        )

        emit_progress(progress, 92, "写入并回读视频元数据")
        metadata_result = inject_metadata(output_path, settings.metadata)
        final_metadata = probe_video(output_path)
        manifest = {
            "schema_version": "reframe-narration-v1",
            "input": {
                "file": video_path.name,
                "path": str(video_path),
                "sha256": sha256_file(video_path),
                **input_metadata,
            },
            "settings": settings.public_dict(),
            "asr": asr_result.to_dict(),
            "llm": rewrite_result.to_dict(),
            "tts": tts_result.to_dict(),
            "render": render_result.to_dict(),
            "metadata": metadata_result.to_dict(),
            "output": {
                "file": output_path.name,
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                **final_metadata,
            },
            "artifacts": {
                "manifest": manifest_path.name,
                "transcript": transcript_path.name,
                "narration": script_path.name,
                "temp_dir": str(work_dir) if keep_temp else None,
            },
        }
        write_json(manifest_path, manifest)
        emit_progress(progress, 100, "处理完成")
        return {
            "status": "completed",
            "input": str(video_path),
            "video": str(output_path),
            "manifest": str(manifest_path),
            "transcript": str(transcript_path),
            "narration": str(script_path),
            "asr_mode": asr_result.mode,
            "llm_mode": rewrite_result.mode,
            "tts_mode": tts_result.mode,
            "metadata_mode": metadata_result.mode,
        }
    except Exception:
        cleanup_paths(created_paths)
        raise
    finally:
        if not keep_temp:
            cleanup_paths([work_dir])
            try:
                work_root.rmdir()
            except OSError:
                pass


def process_batch(
    video_paths: Sequence[Path],
    output_dir: Path,
    settings: AppSettings,
    *,
    fail_fast: bool = False,
    **workflow_options: Any,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    total = len(video_paths)
    for index, video_path in enumerate(video_paths, start=1):
        label = f"[{index}/{total}] {Path(video_path).name}"

        def progress(value: int, message: str, prefix: str = label) -> None:
            print(f"{prefix} | {value:3d}% | {message}")

        try:
            results.append(
                process_video(
                    video_path,
                    output_dir,
                    settings,
                    progress=progress,
                    **workflow_options,
                )
            )
        except Exception as exc:
            failure = {
                "status": "failed",
                "input": str(video_path),
                "error": str(exc),
            }
            results.append(failure)
            if fail_fast:
                break
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI 视频 ASR、文案重构、TTS、画音重构与元数据批处理程序"
    )
    parser.add_argument("inputs", nargs="+", help="视频文件或目录")
    parser.add_argument("-o", "--output-dir", default="output/reconstructed")
    parser.add_argument("--config", type=Path, help="JSON 配置文件")
    parser.add_argument("--recursive", action="store_true", help="递归扫描输入目录")
    parser.add_argument("--default-text", help="ASR 失败或跳过时使用的默认解说文本")
    parser.add_argument("--asr-backend", choices=["auto", "faster-whisper", "whisper"])
    parser.add_argument("--asr-model", help="Whisper 模型名称，默认 base")
    parser.add_argument("--language", help="ASR 语言代码，例如 zh、en")
    parser.add_argument("--api-base", help="OpenAI/DeepSeek 兼容接口地址")
    parser.add_argument("--llm-model", help="文本重构模型名称")
    parser.add_argument("--voice", help="edge-tts 发音人")
    parser.add_argument("--device-make", help="设备厂商元数据")
    parser.add_argument("--device-model", help="设备型号元数据")
    parser.add_argument("--device-software", help="设备软件版本元数据")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _apply_cli_overrides(settings: AppSettings, args: argparse.Namespace) -> None:
    if args.default_text is not None:
        settings.asr = replace(settings.asr, default_text=args.default_text)
    if args.asr_backend:
        settings.asr = replace(settings.asr, backend=args.asr_backend)
    if args.asr_model:
        settings.asr = replace(settings.asr, model=args.asr_model)
    if args.language:
        settings.asr = replace(settings.asr, language=args.language)
    if args.api_base:
        settings.llm = replace(settings.llm, api_base=args.api_base)
    if args.llm_model:
        settings.llm = replace(settings.llm, model=args.llm_model)
    if args.voice:
        settings.tts = replace(settings.tts, voice=args.voice)
    metadata_changes: Dict[str, Any] = {}
    if args.device_make:
        metadata_changes["make"] = args.device_make
    if args.device_model:
        metadata_changes["model"] = args.device_model
    if args.device_software:
        metadata_changes["software"] = args.device_software
    if args.no_metadata:
        metadata_changes["enabled"] = False
    if metadata_changes:
        settings.metadata = replace(settings.metadata, **metadata_changes)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    try:
        settings = load_config(args.config)
        _apply_cli_overrides(settings, args)
        videos = discover_video_files(
            [Path(value) for value in args.inputs],
            recursive=args.recursive,
        )
        if not videos:
            parser.error("没有找到支持的视频文件。")
        results = process_batch(
            videos,
            Path(args.output_dir),
            settings,
            fail_fast=args.fail_fast,
            skip_asr=args.skip_asr,
            skip_llm=args.skip_llm,
            skip_tts=args.skip_tts,
            keep_temp=args.keep_temp,
            overwrite=args.overwrite,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 1 if any(item["status"] == "failed" for item in results) else 0
    except ValueError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
