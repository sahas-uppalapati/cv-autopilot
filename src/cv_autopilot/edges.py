"""Canny stages implemented with NumPy arrays."""

import numpy as np

from .regions import flood_component


def bgr_to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert OpenCV's blue, green, red channel order to luminance."""
    blue, green, red = frame.astype(np.float32).transpose(2, 0, 1)
    return 0.114 * blue + 0.587 * green + 0.299 * red


def gaussian_blur(gray: np.ndarray) -> np.ndarray:
    """Apply the 5-tap binomial approximation in each axis."""
    height, width = gray.shape
    horizontal = np.pad(gray, ((0, 0), (2, 2)), mode="reflect")
    blurred_rows = (
        horizontal[:, :width]
        + 4 * horizontal[:, 1 : width + 1]
        + 6 * horizontal[:, 2 : width + 2]
        + 4 * horizontal[:, 3 : width + 3]
        + horizontal[:, 4:]
    ) / 16
    vertical = np.pad(blurred_rows, ((2, 2), (0, 0)), mode="reflect")
    return (
        vertical[:height]
        + 4 * vertical[1 : height + 1]
        + 6 * vertical[2 : height + 2]
        + 4 * vertical[3 : height + 3]
        + vertical[4:]
    ) / 16


def sobel_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute derivatives with the horizontal and vertical 3×3 Sobel kernels."""
    padded = np.pad(gray, 1, mode="reflect")
    top_left, top, top_right = padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:]
    left, right = padded[1:-1, :-2], padded[1:-1, 2:]
    bottom_left, bottom, bottom_right = padded[2:, :-2], padded[2:, 1:-1], padded[2:, 2:]
    dx = (top_right - top_left) + 2 * (right - left) + (bottom_right - bottom_left)
    dy = (bottom_left - top_left) + 2 * (bottom - top) + (bottom_right - top_right)
    return dx, dy


def thin_edges(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Keep local maxima along one of the four quantized gradient directions."""
    magnitude = np.hypot(dx, dy)
    direction = np.floor((np.degrees(np.arctan2(dy, dx)) + 22.5) / 45).astype(int) % 4
    padded = np.pad(magnitude, 1)
    neighbors = (
        (padded[1:-1, :-2], padded[1:-1, 2:]),  # Left, right
        (padded[:-2, :-2], padded[2:, 2:]),     # Upper left, lower right
        (padded[:-2, 1:-1], padded[2:, 1:-1]), # Up, down
        (padded[:-2, 2:], padded[2:, :-2]),    # Upper right, lower left
    )
    keep = np.zeros_like(magnitude, dtype=bool)
    for angle, (before, after) in enumerate(neighbors):
        # Break equal-magnitude ties consistently across a flat ridge.
        keep |= (direction == angle) & (magnitude > before) & (magnitude >= after)
    keep[[0, -1], :] = False
    keep[:, [0, -1]] = False
    return np.where(keep, magnitude, 0)


def hysteresis(response: np.ndarray, low: int, high: int) -> np.ndarray:
    """Retain strong edges and weak pixels linked to strong ones."""
    candidates = (response >= low) & (response > 0)
    edges = np.zeros_like(candidates)
    for row, col in np.argwhere(response >= high).tolist():
        if candidates[row, col]:
            component = np.asarray(flood_component(candidates, row, col))
            edges[component[:, 0], component[:, 1]] = True
    return edges


def canny_edges(gray: np.ndarray, low: int, high: int, roi: np.ndarray) -> np.ndarray:
    smoothed = gaussian_blur(gray)
    dx, dy = sobel_gradients(smoothed)
    response = thin_edges(dx, dy)
    response[~roi] = 0
    return hysteresis(response, low, high)
