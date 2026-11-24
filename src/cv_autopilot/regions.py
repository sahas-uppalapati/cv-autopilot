"""Eight-neighbor flood fill for binary image regions."""

from collections.abc import Iterator

import numpy as np


def flood_component(mask: np.ndarray, row: int, col: int) -> list[tuple[int, int]]:
    """Visit and clear the true pixels connected to a true seed in mask."""
    height, width = mask.shape
    pending = [(row, col)]
    component = []
    mask[row, col] = False
    while pending:
        row, col = pending.pop()
        component.append((row, col))
        for neighbor_row in range(max(0, row - 1), min(height, row + 2)):
            for neighbor_col in range(max(0, col - 1), min(width, col + 2)):
                if mask[neighbor_row, neighbor_col]:
                    mask[neighbor_row, neighbor_col] = False
                    pending.append((neighbor_row, neighbor_col))
    return component


def connected_components(mask: np.ndarray) -> Iterator[list[tuple[int, int]]]:
    """Yield each component as a list of (row, column) coordinates."""
    remaining = mask.copy()
    for row, col in np.argwhere(remaining).tolist():
        if remaining[row, col]:
            yield flood_component(remaining, row, col)
