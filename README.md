# Mini Lane & Obstacle Detection System

A classical computer vision pipeline for finding lane markings in recorded dashcam footage. See the [project vision](docs/vision.md) for the scope and evaluation plan.

## Setup

Use Python 3.10 or newer. Create a virtual environment with `python -m venv .venv`, then activate it with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in a POSIX shell. Install the project with `python -m pip install -e .`.

Keep recordings and annotations under `data/` or pass absolute paths. Write demo videos under `outputs/`. Both directories are ignored by Git.

## Lane demo

```sh
cv-autopilot lanes data/drive.mp4 outputs/lanes.mp4 --max-frames 120
```

The command reports average processing time per frame, excluding video decoding and encoding. Use these controls to tune the detector.

| Option | What it changes |
| --- | --- |
| `--roi-top` | Moves the top of the road trapezoid up or down as a fraction of frame height. |
| `--roi-width` | Widens or narrows the top edge of the trapezoid. |
| `--canny-low`, `--canny-high` | Set the gradient cutoffs for weak edges and strong edge seeds. |
| `--min-component-pixels` | Drops smaller edge fragments before fitting lines. |

## Lane algorithms

1. Compute grayscale luminance from the blue, green, and red channels.
2. Smooth with a separable Gaussian approximation and calculate Sobel derivatives.
3. Thin gradient ridges with non-maximum suppression, then use two thresholds and eight-neighbor hysteresis inside the road trapezoid.
4. Group connected edge pixels with flood fill and fit each eligible component using the least-squares formula. Select the largest eligible slanted component on each side.

The project code implements these steps with NumPy array operations. OpenCV decodes images and video, draws the lane overlay, and encodes the result. The connected-component fit follows the proposal's fallback. The course requires written approval for the Hough transform.

## Checks

```sh
python -m unittest discover -s tests -v
```

The synthetic checks exercise edge detection, lane fitting, and short-video output without downloading datasets. Use a short local dashcam clip to check that the road region fits your camera.
