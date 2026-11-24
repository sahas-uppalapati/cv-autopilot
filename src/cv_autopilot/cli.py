"""Run lane detection on recorded dashcam footage."""

import argparse
from pathlib import Path

from .lanes import LaneSettings, detect_lanes, draw_lanes
from .video import render_video


def positive_integer(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)

    lanes = commands.add_parser("lanes", help="Render lane boundaries")
    lanes.add_argument("input", type=Path, help="Path to the dashcam clip")
    lanes.add_argument("output", type=Path, help="Path for the annotated MP4")
    lanes.add_argument("--max-frames", type=positive_integer, help="Stop after this many frames")
    lanes.add_argument(
        "--roi-top", type=float, default=LaneSettings.roi_top,
        help="Top of the road trapezoid as a fraction of frame height",
    )
    lanes.add_argument(
        "--roi-width", type=float, default=LaneSettings.roi_width,
        help="Width of the top edge as a fraction of frame width",
    )
    lanes.add_argument(
        "--canny-low", type=int, default=LaneSettings.canny_low,
        help="Minimum gradient magnitude for a connected weak edge",
    )
    lanes.add_argument(
        "--canny-high", type=int, default=LaneSettings.canny_high,
        help="Minimum gradient magnitude for a strong edge seed",
    )
    lanes.add_argument(
        "--min-component-pixels", type=int, default=LaneSettings.min_component_pixels,
        help="Minimum number of connected edge pixels eligible for a line fit",
    )

    args = parser.parse_args(argv)
    try:
        settings = LaneSettings(
            roi_top=args.roi_top,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            min_component_pixels=args.min_component_pixels,
            roi_width=args.roi_width,
        )
        frames, seconds = render_video(
            args.input, args.output,
            lambda frame: draw_lanes(frame, detect_lanes(frame, settings)),
            args.max_frames,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Wrote {frames} frames to {args.output} ({1000 * seconds / frames:.1f} ms/frame)")


if __name__ == "__main__":
    main()
