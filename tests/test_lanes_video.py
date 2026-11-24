import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from cv_autopilot.lanes import LaneSettings, detect_lanes, draw_lanes
from cv_autopilot.video import render_video


def synthetic_road() -> np.ndarray:
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    cv2.line(frame, (32, 119), (88, 74), (255, 255, 255), 3)
    cv2.line(frame, (168, 119), (112, 74), (255, 255, 255), 3)
    return frame


class LaneTests(unittest.TestCase):
    def test_fits_left_and_right_boundaries(self):
        frame = synthetic_road()
        result = detect_lanes(frame, LaneSettings())
        self.assertIsNotNone(result.left)
        self.assertIsNotNone(result.right)
        self.assertLess(abs(result.left[0][0] - 32), 6)
        self.assertLess(abs(result.right[0][0] - 168), 6)
        self.assertFalse(np.array_equal(frame, draw_lanes(frame, result)))
        self.assertTrue(np.array_equal(frame, synthetic_road()))

    def test_blank_frame_has_no_boundaries(self):
        result = detect_lanes(np.zeros_like(synthetic_road()), LaneSettings())
        self.assertIsNone(result.left)
        self.assertIsNone(result.right)

    def test_invalid_thresholds_fail_early(self):
        with self.assertRaises(ValueError):
            LaneSettings(canny_low=200, canny_high=100)


class VideoTests(unittest.TestCase):
    def test_writes_only_requested_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.mp4"
            output = Path(directory) / "nested" / "output.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (200, 120)
            )
            if not writer.isOpened():
                self.skipTest("MP4 encoder unavailable")
            try:
                for _ in range(3):
                    writer.write(synthetic_road())
            finally:
                writer.release()

            count, seconds = render_video(source, output, lambda frame: frame, max_frames=2)
            self.assertEqual(count, 2)
            self.assertGreaterEqual(seconds, 0)
            capture = cv2.VideoCapture(str(output))
            try:
                self.assertTrue(capture.read()[0])
                self.assertTrue(capture.read()[0])
                self.assertFalse(capture.read()[0])
            finally:
                capture.release()

    def test_protects_input_path(self):
        source = Path("clip.mp4")
        with self.assertRaisesRegex(ValueError, "different files"):
            render_video(source, source, lambda frame: frame)
