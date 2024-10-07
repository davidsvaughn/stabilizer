# Video Stabilization Script

Remove jitter from grayscale/IR/thermal drone videos
<p align="center">
  <img src="media/image/orig.gif" width="48%" />
  <img src="media/image/stable.gif" width="48%" /> 
</p>

## Overview

This Python script stabilizes videos by tracking feature points across frames and applying transformations to align the frames. It uses optical flow for tracking and can apply either affine or homography transformations. The result is a smoother, more stable video output.

## Features

- Support for different input and reference videos
- Application of CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhanced tracking
- Shi-Tomasi corner detection for initial feature points
- Lucas-Kanade optical flow for tracking points across frames
- Filtering of optical flow tracks to get the best keypoint pairs
- Option to apply either affine or homography transformations for stabilization
