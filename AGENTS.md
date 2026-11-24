# AGENTS.md

## Project scope

This repository contains the lane detection component of the Mini Lane & Obstacle Detection System. Use TuSimple for annotated lane detection. See [Project Vision](docs/vision.md) for the scope and evaluation plan.

## Vision methods

- Lane detection starts with HSV or grayscale preprocessing, a road region of interest, and Canny edge detection. Use the Hough line transform to fit left and right lane boundaries only if the instructor approves it in writing. The proposed fallback groups edge pixels with connected components and fits lines by least squares.
- Keep the lane algorithms in project code using NumPy for array operations. Use OpenCV for image and video I/O and drawing overlays.
- Keep course techniques central to the implementation. Outside techniques require explicit written instructor approval and can account for at most about 20% of the project.

## Implementation and verification

- Keep the lane demo and evaluation independently runnable where practical. Make dataset paths and algorithm parameters configurable. Keep raw datasets and generated videos out of version control.
- Keep setup and lane demo commands current in the README. Verify changes on a short representative clip before running a larger evaluation.
- Measure lane precision and recall against annotated frames and report processing time per frame. Record the dataset split, matching criteria, and run settings so results can be reproduced. Compare results across lighting and traffic conditions.
