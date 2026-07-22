import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asr_module import transcribe_video
from config import (
    AppSettings,
    ASRSettings,
    LLMSettings,
    MetadataSettings,
    RenderSettings,
    TTSSettings,
    load_config,
)
from llm_module import rewrite_narration
from main import process_batch, process_video
from metadata_injector import inject_metadata
from tts_module import synthesize_speech
from utils import discover_video_files, probe_video
from video_renderer import render_video


class ConfigAndFallbackTests(unittest.TestCase):
    def test_nested_json_config_and_environment_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "asr": {"model": "base", "default_text": "备用文案"},
                        "llm": {
                            "api_base": "https://api.deepseek.com/v1",
                            "model": "deepseek-chat",
                        },
                        "tts": {"voice": "zh-CN-YunxiNeural"},
                        "metadata": {"enabled": True, "make": "Xiaomi"},
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"VIDEO_LLM_API_KEY": "env-key"}, clear=False):
                settings = load_config(path)

        self.assertEqual(settings.asr.model, "base")
        self.assertEqual(settings.llm.model, "deepseek-chat")
        self.assertEqual(settings.llm.api_key, "env-key")
        self.assertEqual(settings.tts.voice, "zh-CN-YunxiNeural")
        self.assertEqual(settings.metadata.make, "Xiaomi")

    def test_asr_uses_default_text_when_audio_or_whisper_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "input.mp4"
            video.write_bytes(b"not-a-real-video")
            with patch("asr_module.extract_audio", side_effect=RuntimeError("ffmpeg failed")):
                result = transcribe_video(
                    video,
                    root / "work",
                    ASRSettings(default_text="这里是备用解说"),
                )

        self.assertEqual(result.text, "这里是备用解说")
        self.assertEqual(result.mode, "default-text")
        self.assertTrue(result.warning)

    def test_llm_without_key_keeps_transcript_as_safe_fallback(self):
        result = rewrite_narration(
            "我拿起这个产品开始使用。",
            LLMSettings(api_key=None),
        )

        self.assertEqual(result.text, "我拿起这个产品开始使用。")
        self.assertEqual(result.mode, "source-fallback")

    def test_tts_missing_dependency_skips_without_breaking_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "voice.mp3"
            with patch.dict(sys.modules, {"edge_tts": None}):
                result = synthesize_speech(
                    "新的旁白",
                    output,
                    TTSSettings(voice="zh-CN-YunxiNeural"),
                )

        self.assertIsNone(result.audio_path)
        self.assertEqual(result.mode, "skipped")
        self.assertTrue(result.warning)

    def test_missing_exiftool_is_a_non_fatal_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "output.mp4"
            video.write_bytes(b"placeholder")
            original_which = shutil.which
            with patch(
                "metadata_injector.shutil.which",
                side_effect=lambda name: (
                    None if name == "exiftool" else original_which(name)
                ),
            ):
                result = inject_metadata(
                    video,
                    MetadataSettings(enabled=True, make="Apple", model="iPhone 15 Pro"),
                )

        self.assertFalse(result.applied)
        self.assertEqual(result.mode, "skipped")
        self.assertIn("ExifTool", result.warning)

    def test_video_discovery_supports_recursive_batch_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "nested"
            nested.mkdir()
            (root / "first.mp4").write_bytes(b"one")
            (nested / "second.mov").write_bytes(b"two")
            (nested / "ignore.txt").write_text("ignore", encoding="utf-8")

            shallow = discover_video_files([root], recursive=False)
            recursive = discover_video_files([root], recursive=True)

        self.assertEqual([path.name for path in shallow], ["first.mp4"])
        self.assertEqual(
            sorted(path.name for path in recursive),
            ["first.mp4", "second.mov"],
        )

    def test_batch_continues_after_a_single_file_failure(self):
        settings = AppSettings(metadata=MetadataSettings(enabled=False))
        with patch(
            "main.process_video",
            side_effect=[RuntimeError("broken input"), {"status": "completed"}],
        ) as process:
            results = process_batch(
                [Path("broken.mp4"), Path("working.mp4")],
                Path("outputs"),
                settings,
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["status"], "failed")
        self.assertEqual(results[1]["status"], "completed")
        self.assertEqual(process.call_count, 2)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class RendererTests(unittest.TestCase):
    def test_renderer_replaces_original_audio_and_outputs_vertical_h264(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            narration = root / "narration.wav"
            output = root / "rendered.mp4"
            subprocess.run(
                [
                    shutil.which("ffmpeg"),
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=320x180:rate=24:duration=2",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=220:sample_rate=48000:duration=2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(source),
                ],
                check=True,
            )
            subprocess.run(
                [
                    shutil.which("ffmpeg"),
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=880:sample_rate=48000:duration=1.5",
                    str(narration),
                ],
                check=True,
            )

            result = render_video(
                source,
                narration,
                output,
                RenderSettings(width=360, height=640, preset="ultrafast", crf=28),
            )
            metadata = probe_video(output)

        self.assertTrue(result.output_path.name.endswith(".mp4"))
        self.assertTrue(result.used_narration)
        self.assertEqual((metadata["width"], metadata["height"]), (360, 640))
        self.assertEqual(metadata["video_codec"], "h264")
        self.assertEqual(metadata["audio_codec"], "aac")
        self.assertGreater(metadata["duration"], 1.2)
        self.assertLess(metadata["duration"], 1.8)

    def test_complete_workflow_survives_disabled_external_adapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "workflow-source.mp4"
            subprocess.run(
                [
                    shutil.which("ffmpeg"),
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=320x180:rate=24:duration=1.2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(source),
                ],
                check=True,
            )
            settings = AppSettings(
                asr=ASRSettings(default_text="这是新的解说词"),
                llm=LLMSettings(api_key=None),
                tts=TTSSettings(),
                render=RenderSettings(
                    width=360,
                    height=640,
                    preset="ultrafast",
                    crf=28,
                ),
                metadata=MetadataSettings(enabled=False),
            )

            result = process_video(
                source,
                root / "outputs",
                settings,
                skip_asr=True,
                skip_tts=True,
            )
            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["asr_mode"], "manual-text")
        self.assertEqual(result["llm_mode"], "source-fallback")
        self.assertEqual(result["tts_mode"], "disabled")
        self.assertEqual(result["metadata_mode"], "disabled")
        self.assertEqual(manifest["output"]["width"], 360)
        self.assertEqual(manifest["output"]["height"], 640)
        self.assertEqual(manifest["output"]["audio_codec"], "aac")

    def test_failed_overwrite_preserves_previous_completed_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            subprocess.run(
                [
                    shutil.which("ffmpeg"),
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=blue:size=160x90:rate=24:duration=0.8",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(source),
                ],
                check=True,
            )
            settings = AppSettings(
                asr=ASRSettings(default_text="测试"),
                llm=LLMSettings(api_key=None),
                render=RenderSettings(
                    width=180,
                    height=320,
                    preset="ultrafast",
                    crf=30,
                ),
                metadata=MetadataSettings(enabled=False),
            )
            first = process_video(
                source,
                root / "outputs",
                settings,
                skip_asr=True,
                skip_tts=True,
            )
            output = Path(first["video"])
            previous = output.read_bytes()

            with patch("main.render_video", side_effect=RuntimeError("render failed")):
                with self.assertRaises(RuntimeError):
                    process_video(
                        source,
                        root / "outputs",
                        settings,
                        skip_asr=True,
                        skip_tts=True,
                        overwrite=True,
                    )

            self.assertTrue(output.exists())
            self.assertEqual(output.read_bytes(), previous)

    def test_invalid_video_does_not_leave_temporary_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "invalid.mp4"
            source.write_bytes(b"invalid video")
            output_dir = root / "outputs"

            with self.assertRaises(Exception):
                process_video(
                    source,
                    output_dir,
                    AppSettings(metadata=MetadataSettings(enabled=False)),
                )

            self.assertFalse((output_dir / ".reframe-work").exists())

    @unittest.skipUnless(shutil.which("exiftool"), "ExifTool required")
    def test_custom_handler_and_phone_fields_are_verified_by_readback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "metadata-source.mp4"
            subprocess.run(
                [
                    shutil.which("ffmpeg"),
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=green:size=160x90:rate=24:duration=0.8",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(source),
                ],
                check=True,
            )
            settings = AppSettings(
                asr=ASRSettings(default_text="测试"),
                llm=LLMSettings(api_key=None),
                render=RenderSettings(
                    width=180,
                    height=320,
                    preset="ultrafast",
                    crf=30,
                ),
                metadata=MetadataSettings(
                    enabled=True,
                    make="Xiaomi",
                    model="Xiaomi 14 Pro",
                    software="HyperOS 2",
                    handler_description="Custom Core Media Video",
                ),
            )

            result = process_video(
                source,
                root / "outputs",
                settings,
                skip_asr=True,
                skip_tts=True,
            )
            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            readback = manifest["metadata"]["readback"]

        self.assertTrue(manifest["metadata"]["applied"])
        self.assertEqual(readback["Keys:Make"], "Xiaomi")
        self.assertEqual(readback["Keys:Model"], "Xiaomi 14 Pro")
        self.assertIn(
            "Custom Core Media Video",
            [
                value
                for key, value in readback.items()
                if key.endswith(":HandlerDescription")
            ],
        )

    def test_missing_exiftool_warning_is_emitted_by_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "warning-source.mp4"
            subprocess.run(
                [
                    shutil.which("ffmpeg"),
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=red:size=160x90:rate=24:duration=0.6",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(source),
                ],
                check=True,
            )
            settings = AppSettings(
                asr=ASRSettings(default_text="测试"),
                llm=LLMSettings(api_key=None),
                render=RenderSettings(
                    width=180,
                    height=320,
                    preset="ultrafast",
                    crf=30,
                ),
                metadata=MetadataSettings(enabled=True),
            )

            original_which = shutil.which
            with patch(
                "metadata_injector.shutil.which",
                side_effect=lambda name: (
                    None if name == "exiftool" else original_which(name)
                ),
            ):
                with self.assertLogs("reframe", level="WARNING") as captured:
                    result = process_video(
                        source,
                        root / "outputs",
                        settings,
                        skip_asr=True,
                        skip_tts=True,
                    )

        self.assertEqual(result["metadata_mode"], "skipped")
        self.assertIn("ExifTool", "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()
