import tempfile
import unittest
from pathlib import Path

from pipeline import build_ass, create_fallback_plan, normalize_plan


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


if __name__ == "__main__":
    unittest.main()
