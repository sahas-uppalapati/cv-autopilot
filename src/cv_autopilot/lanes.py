"""Lane detection using the connected-component fallback in the proposal."""

from dataclasses import dataclass

import cv2
import numpy as np

from .edges import bgr_to_grayscale, canny_edges
from .regions import connected_components

Point = tuple[int, int]
Lane = tuple[Point, Point]


@dataclass(frozen=True)
class LaneSettings:
    roi_top: float = 0.60
    canny_low: int = 50
    canny_high: int = 150
    min_component_pixels: int = 15
    roi_width: float = 0.14

    def __post_init__(self) -> None:
        if not 0 < self.roi_top < 1:
            raise ValueError("ROI top must be between 0 and 1")
        if not 0 < self.roi_width <= 1:
            raise ValueError("ROI width must be between 0 (exclusive) and 1")
        if not 0 <= self.canny_low < self.canny_high <= 255:
            raise ValueError("Canny thresholds must be ordered within 0–255")
        if self.min_component_pixels < 2:
            raise ValueError("Each component needs at least two pixels")


@dataclass(frozen=True)
class LaneDetection:
    left: Lane | None = None
    right: Lane | None = None


def road_region(height: int, width: int, settings: LaneSettings) -> tuple[np.ndarray, int]:
    """Return a centered trapezoid that widens to the frame edges at the bottom."""
    top = min(round(height * settings.roi_top), height - 1)
    rows = np.arange(height)[:, None]
    columns = np.arange(width)
    center = (width - 1) / 2
    progress = (rows - top) / max(height - 1 - top, 1)
    half_width = center * (settings.roi_width + (1 - settings.roi_width) * progress)
    half_width[-1] = center
    return (rows >= top) & (abs(columns - center) <= half_width), top


def road_edges(frame: np.ndarray, settings: LaneSettings) -> tuple[np.ndarray, int]:
    """Compute edge responses and follow connected edges within the road region."""
    roi, top = road_region(*frame.shape[:2], settings)
    gray = bgr_to_grayscale(frame)
    return canny_edges(gray, settings.canny_low, settings.canny_high, roi), top


def fit_line(rows: np.ndarray, columns: np.ndarray) -> tuple[float, float]:
    """Fit x = slope * y + offset using the closed-form least-squares solution."""
    mean_y, mean_x = rows.mean(), columns.mean()
    centered_y = rows - mean_y
    slope = np.sum(centered_y * (columns - mean_x)) / np.sum(centered_y ** 2)
    return float(slope), float(mean_x - slope * mean_y)


def detect_lanes(frame: np.ndarray, settings: LaneSettings) -> LaneDetection:
    """Fit the strongest slanted component on each side as x = slope * y + offset."""
    edges, top = road_edges(frame, settings)
    height, width = edges.shape
    best: dict[str, tuple[int, Lane]] = {}

    for component in connected_components(edges):
        area = len(component)
        if area < settings.min_component_pixels:
            continue
        rows, columns = np.asarray(component).T
        if rows.max() - rows.min() < 3:
            continue
        slope, offset = fit_line(rows, columns)
        if not 0.15 <= abs(slope) <= 4.0:
            continue

        bottom_x = round(slope * (height - 1) + offset)
        is_left = slope < 0 and bottom_x < width / 2
        is_right = slope > 0 and bottom_x > width / 2
        if not (is_left or is_right) or not 0 <= bottom_x < width:
            continue
        side = "left" if is_left else "right"
        if side in best and area <= best[side][0]:
            continue
        top_x = int(np.clip(round(slope * top + offset), 0, width - 1))
        best[side] = (area, ((bottom_x, height - 1), (top_x, top)))

    return LaneDetection(
        left=best["left"][1] if "left" in best else None,
        right=best["right"][1] if "right" in best else None,
    )


def draw_lanes(frame: np.ndarray, detection: LaneDetection) -> np.ndarray:
    annotated = frame.copy()
    for lane in (detection.left, detection.right):
        if lane is not None:
            cv2.line(annotated, *lane, (50, 230, 50), 3, cv2.LINE_AA)
    return annotated
