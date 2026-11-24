"""Read dashcam footage and render annotated clips."""

from collections.abc import Callable, Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np

FrameProcessor = Callable[[np.ndarray], np.ndarray]


def video_frames(capture: cv2.VideoCapture) -> Iterator[np.ndarray]:
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        yield frame


def render_video(
    source: Path,
    destination: Path,
    process: FrameProcessor,
    max_frames: int | None = None,
) -> tuple[int, float]:
    """Write annotated MP4 frames and return count and processing time."""
    if max_frames is not None and max_frames < 1:
        raise ValueError("Frame limit must be positive")
    if source.resolve() == destination.resolve() or (
        destination.exists() and source.exists() and source.samefile(destination)
    ):
        raise ValueError("Input and output must be different files")
    if destination.suffix.lower() != ".mp4":
        raise ValueError("Output path must end in .mp4")

    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Unable to read video frame rate: {source}")

        frames = video_frames(capture)
        first = next(frames, None)
        if first is None:
            raise ValueError(f"No decodable frames in {source}")
        height, width = first.shape[:2]
        destination.parent.mkdir(parents=True, exist_ok=True)

        with TemporaryDirectory(dir=destination.parent) as directory:
            temporary = Path(directory) / "render.mp4"
            writer = cv2.VideoWriter(
                str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            count = 0
            seconds = 0.0
            try:
                if not writer.isOpened():
                    raise ValueError(f"Unable to write MP4 under {destination.parent}")
                frame = first
                while frame is not None:
                    start = perf_counter()
                    annotated = process(frame)
                    seconds += perf_counter() - start
                    writer.write(annotated)
                    count += 1
                    if max_frames is not None and count >= max_frames:
                        break
                    frame = next(frames, None)
            finally:
                writer.release()

            temporary.replace(destination)
        return count, seconds
    finally:
        capture.release()
