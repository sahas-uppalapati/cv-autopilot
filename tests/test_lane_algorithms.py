import unittest

import numpy as np

from cv_autopilot.edges import (
    bgr_to_grayscale,
    canny_edges,
    gaussian_blur,
    hysteresis,
    sobel_gradients,
)
from cv_autopilot.lanes import LaneSettings, fit_line, road_edges, road_region
from cv_autopilot.regions import connected_components


class EdgeTests(unittest.TestCase):
    def test_grayscale_weights_bgr_channels(self):
        frame = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        np.testing.assert_allclose(bgr_to_grayscale(frame), [[29.07, 149.685, 76.245]])

    def test_gaussian_blur_preserves_constant_intensity_and_impulse_sum(self):
        constant = np.full((5, 7), 25.0)
        np.testing.assert_array_equal(gaussian_blur(constant), constant)
        impulse = np.zeros((9, 9))
        impulse[4, 4] = 1
        blurred = gaussian_blur(impulse)
        self.assertAlmostEqual(blurred.sum(), 1)
        self.assertAlmostEqual(blurred[4, 4], (6 / 16) ** 2)

    def test_sobel_gradient_points_across_an_intensity_step(self):
        gray = np.zeros((7, 9))
        gray[:, 4:] = 255
        dx, dy = sobel_gradients(gray)
        np.testing.assert_array_equal(dy, np.zeros_like(gray))
        self.assertEqual(dx[3, 3], 4 * 255)
        self.assertEqual(dx[3, 1], 0)

    def test_canny_keeps_one_pixel_per_row_on_a_step(self):
        gray = np.zeros((20, 20))
        gray[:, 10:] = 255
        roi = np.ones_like(gray, dtype=bool)
        edges = canny_edges(gray, low=50, high=150, roi=roi)
        np.testing.assert_array_equal(edges[1:-1].sum(axis=1), np.ones(18))
        self.assertFalse(edges[:, :9].any())
        self.assertFalse(edges[:, 11:].any())

    def test_hysteresis_preserves_linked_weak_edges_only(self):
        response = np.zeros((6, 6))
        response[1, 1] = 200
        response[2, 2] = response[3, 3] = 80
        response[0, 5] = 80
        self.assertCountEqual(
            np.argwhere(hysteresis(response, low=50, high=150)).tolist(),
            [[1, 1], [2, 2], [3, 3]],
        )


class LaneGeometryTests(unittest.TestCase):
    def test_components_include_diagonals_and_keep_groups_separate(self):
        mask = np.array([[True, False, False, True], [False, True, False, False]])
        components = list(connected_components(mask))
        self.assertEqual([set(component) for component in components], [{(0, 0), (1, 1)}, {(0, 3)}])
        self.assertEqual(mask.sum(), 3)

    def test_line_fit_recovers_slope_and_offset(self):
        rows = np.array([10, 12, 14, 16])
        columns = 35 - 2 * rows
        self.assertEqual(fit_line(rows, columns), (-2.0, 35.0))

    def test_roi_bounds_and_top_width(self):
        narrow, top = road_region(30, 40, LaneSettings(roi_width=0.2))
        wide, _ = road_region(30, 40, LaneSettings(roi_width=0.6))
        self.assertFalse(narrow[:top].any())
        self.assertLess(narrow[top].sum(), wide[top].sum())
        self.assertTrue(narrow[-1].all())

    def test_road_edges_are_excluded_outside_the_roi(self):
        frame = np.zeros((30, 40, 3), dtype=np.uint8)
        frame[:, 20:] = 255
        settings = LaneSettings()
        edges, top = road_edges(frame, settings)
        self.assertFalse(edges[:top].any())
        self.assertTrue(edges[top:].any())
