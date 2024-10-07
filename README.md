# Video Stabilization Script

Remove jitter from grayscale/IR/thermal drone videos
<p align="center">
  <img src="media/image/orig.gif" width="48%" />
  <img src="media/image/stable.gif" width="48%" /> 
</p>

## Overview

This Python script stabilizes videos by tracking feature points across frames and applying transformations to align the frames. It uses optical flow for tracking and can apply either affine or homography transformations.

## Features

- Support for different input and reference videos (tracking computed on ref video, applied to input video)
- Application of CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhanced tracking
- Shi-Tomasi corner detection for initial feature points
- Lucas-Kanade optical flow for tracking points across frames
- Filtering of optical flow tracks to get the best keypoint pairs
- Option to apply either affine or homography transformations for stabilization

## Usage

Run the script from the command line with the following syntax:
```
python stabilize.py --input <input_video> [options]
```
### Command-line Options:

- `--input`: Path of input video (required)
- `--ref`: Path of reference video for tracking (optional, defaults to input video)
- `--output`: Path of output video (optional, auto-generated if not provided)
- `--trans`: Transform type (0=Affine, 1=Homography, default=0)
- `--min_features`: Minimum number of Shi-Tomasi corners (default=60)
- `--center`: Frame number to use as center reference frame (-1 triggers auto-search, default=-1)
- `--clahe`: Apply CLAHE to output (default=8, set to 0 to disable)
- `--crop/--no-crop`: Enable/disable cropping of output video to remove borders (default: --crop)
