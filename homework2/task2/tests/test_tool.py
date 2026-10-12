"""Verify that the registered tool returns actual saved camera evidence."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from focusing_tool import make_focusing_tool
from test_bench import settings


class ToolTests(unittest.TestCase):
    def test_registered_tool_saves_measurements_and_emits_actual_events(self):
        events = []
        with tempfile.TemporaryDirectory() as root:
            tool = make_focusing_tool(root, event=lambda kind, data: events.append((kind, data)))
            result = tool.invoke(settings(iterations=10).model_dump())
            folder = Path(result["working_directory"])
            self.assertEqual(tool.name, "focus_beam")
            self.assertTrue(result["verified"])
            self.assertAlmostEqual(result["final"]["enhancement"],
                                   np.mean(result["final"]["shot_enhancements"]))
            self.assertEqual(np.load(folder / "final_image.npy").shape, (32, 32))
            self.assertEqual(json.loads((folder / "result.json").read_text()), result)
            self.assertTrue(all(Path(path).is_file() for path in result["camera_images"].values()))
            self.assertEqual(events[0][0], "tool_started")
            self.assertEqual(events[-1][0], "tool_completed")


if __name__ == "__main__":
    unittest.main()
