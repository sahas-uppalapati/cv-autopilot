# Project Vision

## Goal

Build a lane detector that takes a recorded dashcam clip and draws the left and right lane boundaries on the video. The first version focuses on visible markings on straight roads. It should make each step easy to inspect so we can tell why a frame succeeds or fails.

## Scope

- Process ordinary dashcam videos one frame at a time and save an annotated MP4.
- Let the user adjust the road region, edge thresholds, and minimum size of a connected edge group.
- Report the average processing time per frame.
- Keep the project focused on lane geometry in recorded footage. Vehicle detection, tracking, and driving controls are outside this repository's scope.

## Approach

1. Convert each frame to grayscale and smooth it with a separable Gaussian approximation.
2. Calculate Sobel gradients, thin the gradient ridges, and use Canny hysteresis to keep connected edges inside the road region.
3. Group connected edge pixels. Fit each eligible group with least squares and select a boundary on each side based on its slope and position.
4. Draw the detected boundaries on the original frame and encode the annotated video.

The detector implements these steps directly with NumPy. OpenCV reads the input, draws the overlay, and writes the output. This baseline uses straight-line fits. Curves, gaps in the paint, and shadows may cause it to lose a boundary or pick the wrong edge.

## Evaluation plan

Use the Udacity clips for a visual check and the TuSimple annotations for measured results. [Udacity sample videos](https://github.com/udacity/CarND-LaneLines-P1/tree/master/test_videos) · [TuSimple lane benchmark](https://github.com/TuSimple/tusimple-benchmark)

- Start with a short clip and check whether the overlays follow the visible lane markings.
- Use the same matching rule and data split across runs. Report lane precision, recall, and average processing time per frame.
- Compare clear roads with clips that have shadows, dashed or faded markings, and heavier traffic.
- Record examples where the detector misses a line or follows a nearby road edge. Keep the clip and settings with each example so it can be reproduced.

The project currently supports the video demo and synthetic tests. Quantitative evaluation against TuSimple annotations is planned.

## What a working baseline looks like

A fresh checkout can run the detector on a short clip using the [README](../README.md). Both boundaries should follow visible markings on representative straight-road footage. When a boundary is missed, the annotated video and run settings should give us enough information to reproduce the miss and see which stage needs work.
