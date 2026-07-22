import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import (
    build_ass,
    create_fallback_plan,
    normalize_plan,
    probe_video,
    public_ai_error,
    run_pipeline,
    select_ai_candidates,
)


def sample_analyses():
    return [
        {
            "source_id": "source_0",
            "path": "/tmp/source_0.mp4",
            "has_audio": True,
            "segments": [
                {
                    "source_id": "source_0",
                    "start": 0.0,
                    "end": 2.8,
                    "motion_score": 42.0,
                    "focus_x": 0.25,
                    "focus_y": 0.45,
                },
                {
                    "source_id": "source_0",
                    "start": 2.8,
                    "end": 6.0,
                    "motion_score": 20.0,
                    "focus_x": 0.7,
                    "focus_y": 0.5,
                },
            ],
        },
        {
            "source_id": "source_1",
            "path": "/tmp/source_1.mp4",
            "has_audio": False,
            "segments": [
                {
                    "source_id": "source_1",
                    "start": 0.0,
                    "end": 3.4,
                    "motion_score": 36.0,
                    "focus_x": 0.5,
                    "focus_y": 0.4,
                },
                {
                    "source_id": "source_1",
                    "start": 3.4,
                    "end": 7.0,
                    "motion_score": 18.0,
                    "focus_x": 0.45,
                    "focus_y": 0.6,
                },
            ],
        },
    ]


class PlanningTests(unittest.TestCase):
    def test_ai_error_redacts_credentials(self):
        error = RuntimeError("401 invalid_api_key sk-example-secret")

        message = public_ai_error(error)

        self.assertNotIn("sk-example-secret", message)
        self.assertIn("401", message)

    def test_fallback_plan_is_renderable_and_uses_multiple_sources(self):
        plan = create_fallback_plan(
            sample_analyses(),
            target_duration=8.0,
            brief="前三秒展示变化。强调轻便、易用。最后给出行动提示。",
            style="product-demo",
        )

        self.assertGreaterEqual(len(plan["beats"]), 3)
        self.assertGreaterEqual(len({beat["source_id"] for beat in plan["beats"]}), 2)
        self.assertLessEqual(plan["estimated_duration"], 8.05)
        self.assertTrue(plan["hook_text"])

    def test_normalize_plan_discards_unknown_sources_and_clamps_ranges(self):
        raw = {
            "title": "AI plan",
            "beats": [
                {
                    "source_id": "missing",
                    "start": 0,
                    "end": 10,
                    "caption": "discard me",
                },
                {
                    "source_id": "source_0",
                    "start": -5,
                    "end": 99,
                    "speed": 9,
                    "focus_x": -2,
                    "focus_y": 4,
                    "caption": "A valid beat",
                },
            ],
        }

        plan = normalize_plan(raw, sample_analyses(), target_duration=4.0, brief="demo")

        self.assertTrue(plan["beats"])
        first = plan["beats"][0]
        self.assertEqual(first["source_id"], "source_0")
        self.assertGreaterEqual(first["start"], 0)
        self.assertLessEqual(first["end"], 6.0)
        self.assertGreaterEqual(first["speed"], 0.75)
        self.assertLessEqual(first["speed"], 1.35)
        self.assertGreaterEqual(first["focus_x"], 0)
        self.assertLessEqual(first["focus_x"], 1)
        self.assertLessEqual(plan["estimated_duration"], 4.05)

    def test_normalize_plan_rejects_timestamp_outside_detected_candidates(self):
        analyses = sample_analyses()
        analyses[0]["duration"] = 8.0
        analyses[0]["segments"] = [
            {**analyses[0]["segments"][0], "start": 0.0, "end": 2.0},
            {**analyses[0]["segments"][1], "start": 5.0, "end": 8.0},
        ]
        raw = {
            "mode": "ai-vision-plan",
            "beats": [
                {
                    "source_id": "source_0",
                    "start": 2.5,
                    "end": 4.5,
                    "caption": "hallucinated range",
                }
            ],
        }

        plan = normalize_plan(raw, analyses, target_duration=4.0, brief="fallback")

        self.assertEqual(plan["mode"], "local-smart-plan")
        self.assertFalse(
            any(
                beat["source_id"] == "source_0"
                and beat["start"] < 4.5
                and beat["end"] > 2.5
                for beat in plan["beats"]
            )
        )

    def test_ai_candidate_scope_contains_only_shared_top_segments(self):
        analyses = sample_analyses()[:1]
        analyses[0]["segments"] = [
            {
                "source_id": "source_0",
                "start": float(index),
                "end": float(index + 1),
                "motion_score": float(index),
            }
            for index in range(13)
        ]

        candidates, restricted = select_ai_candidates(analyses, limit=12)

        self.assertEqual(len(candidates), 12)
        self.assertEqual(len(restricted[0]["segments"]), 12)
        self.assertFalse(any(segment["start"] == 0.0 for segment in candidates))
        self.assertFalse(
            any(segment["start"] == 0.0 for segment in restricted[0]["segments"])
        )

    def test_subclip_focus_track_is_interpolated_to_its_own_range(self):
        analyses = sample_analyses()[:1]
        analyses[0]["segments"] = [
            {
                "source_id": "source_0",
                "start": 0.0,
                "end": 4.0,
                "motion_score": 10.0,
                "focus_x": 0.5,
                "focus_y": 0.5,
                "focus_start_x": 0.0,
                "focus_start_y": 0.2,
                "focus_end_x": 1.0,
                "focus_end_y": 0.8,
            }
        ]
        raw = {
            "beats": [
                {
                    "source_id": "source_0",
                    "start": 1.0,
                    "end": 2.0,
                    "focus_x": 0.5,
                    "focus_y": 0.5,
                }
            ]
        }

        plan = normalize_plan(raw, analyses, target_duration=3.0, brief="focus")

        self.assertAlmostEqual(plan["beats"][0]["focus_start_x"], 0.25, places=2)
        self.assertAlmostEqual(plan["beats"][0]["focus_end_x"], 0.5, places=2)

    def test_ass_file_contains_timeline_and_escaped_caption(self):
        plan = {
            "beats": [
                {
                    "source_id": "source_0",
                    "start": 0,
                    "end": 2,
                    "speed": 1,
                    "caption": "Hook {now}",
                },
                {
                    "source_id": "source_1",
                    "start": 0,
                    "end": 3,
                    "speed": 1,
                    "caption": "Second line",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "captions.ass"
            build_ass(plan, path, width=1080, height=1920)
            content = path.read_text(encoding="utf-8-sig")

        self.assertIn("PlayResX: 1080", content)
        self.assertIn("PlayResY: 1920", content)
        self.assertIn("Hook \\{now\\}", content)
        self.assertIn("0:00:02.00", content)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class RenderSmokeTests(unittest.TestCase):
    def test_full_render_with_mixed_audio_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.mp4"
            second = root / "second.mp4"
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
                    "sine=frequency=440:sample_rate=48000:duration=2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(first),
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
                    "color=c=0x315efb:size=180x320:rate=30:duration=2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(second),
                ],
                check=True,
            )

            result = run_pipeline(
                [first, second],
                root / "output",
                brief="先展示变化；再展示细节；最后行动提示",
                target_duration=3.0,
                style="fast-cut",
                use_ai=False,
                width=360,
                height=640,
            )
            output = root / "output" / result["video"]
            metadata = probe_video(output)

            self.assertTrue(output.exists())
            self.assertEqual((metadata["width"], metadata["height"]), (360, 640))
            self.assertEqual(metadata["video_codec"], "h264")
            self.assertEqual(metadata["audio_codec"], "aac")
            self.assertGreater(metadata["duration"], 2.5)

            mocked_plan = {
                "title": "Mock AI plan",
                "creative_angle": "结果钩子后接细节证明",
                "hook_text": "先看结果",
                "mode": "ai-vision-plan",
                "beats": [
                    {
                        "source_id": "source_0",
                        "start": 0.0,
                        "end": 1.0,
                        "purpose": "hook",
                        "caption": "先看结果",
                    },
                    {
                        "source_id": "source_1",
                        "start": 0.0,
                        "end": 1.0,
                        "purpose": "proof",
                        "caption": "再看细节",
                    },
                    {
                        "source_id": "source_0",
                        "start": 1.0,
                        "end": 2.0,
                        "purpose": "cta",
                        "caption": "立即行动",
                    },
                ],
            }
            with patch("pipeline.create_ai_plan", return_value=mocked_plan):
                ai_result = run_pipeline(
                    [first, second],
                    root / "ai-output",
                    brief="AI path",
                    target_duration=3.0,
                    style="fast-cut",
                    use_ai=True,
                    width=360,
                    height=640,
                )

            self.assertEqual(ai_result["planning_mode"], "ai-vision-plan")
            self.assertIsNone(ai_result["ai_error"])
            self.assertGreater(
                ai_result["transformation"]["visual_change"]["sample_count"], 0
            )


if __name__ == "__main__":
    unittest.main()
