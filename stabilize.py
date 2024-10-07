"""
Video Stabilization Script

This script stabilizes a grayscale video by tracking feature points across frames and applying
transformations to align the frames. It uses optical flow for tracking and can apply either 
affine or homography transformations.

Key features:
1. Supports different input and reference videos
2. Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhanced tracking
3. Uses Shi-Tomasi corner detection for initial feature points
4. Implements Lucas-Kanade optical flow for tracking points across frames
5. Filters optical flow tracks to get the best keypoint pairs
6. Applies either affine or homography transformations for stabilization
7. Optionally crops the output video to remove black borders
8. Matches histograms of transformed frames to maintain consistent appearance

Usage:
python script_name.py --input <input_video> [options]

For full list of options, use: python script_name.py --help

Dependencies:
- OpenCV (cv2)
- NumPy
- NetworkX
- similaritymeasures
- av (PyAV)
- scikit-image

Author: David Vaughn
Date: 2023-09-30
Version: 1.0
"""

import os
import sys
import ntpath
import numpy as np
from glob import glob
import cv2 as cv
import argparse
import networkx as nx
import similaritymeasures as sm
import av
from skimage.exposure import match_histograms


def init_clahe(cliplimit=3.0, dim=8):
    """
    Initialize CLAHE (Contrast Limited Adaptive Histogram Equalization) object.
    
    Args:
    cliplimit (float): Threshold for contrast limiting
    dim (int): Size of grid for histogram equalization
    
    Returns:
    cv.CLAHE: Initialized CLAHE object
    """
    return cv.createCLAHE(clipLimit=cliplimit, tileGridSize=(dim, dim))

def apply_clahe(clahe, img):
    """
    Apply CLAHE to an image.
    
    Args:
    clahe (cv.CLAHE): CLAHE object
    img (np.array): Input image
    
    Returns:
    np.array: CLAHE-enhanced image
    """
    if clahe is None:
        return img
    if len(img.shape) == 3:
        return clahe.apply(img[:, :, 0].astype(np.uint8))
    else:
        return clahe.apply(img.astype(np.uint8))

def read_frames(video_path, clahe=[]):
    """
    Read frames from a video file and optionally apply CLAHE.
    
    Args:
    video_path (str): Path to the video file
    clahe (list): List of dimensions for CLAHE application
    
    Returns:
    tuple: List of frames and video FPS
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = int(stream.average_rate)
    frames = []
    cc = [init_clahe(dim=d) for d in clahe]

    for frame in container.decode(video=0):
        frame = np.array(frame.to_image())[:, :, 0]

        for c in cc: 
            frame = apply_clahe(c, frame)

        frames.append(frame)
    return frames, fps

def write_frames(frames, video_path, fps):
    """
    Write frames to a video file.
    
    Args:
    frames (list): List of frames to write
    video_path (str): Path to save the video
    fps (int): Frames per second for the output video
    """
    container = av.open(video_path, mode='w')
    stream = container.add_stream('mpeg4', rate=fps)
    stream.width = frames[0].shape[1]
    stream.height = frames[0].shape[0]
    stream.pix_fmt = 'yuv420p'

    for frame in frames:
        frame = (frame[:, :, None] * np.ones(3, dtype=int)[None, None, :]).astype(np.uint8)
        frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()

def rotate(p, degrees=0, origin=(0, 0)):
    """
    Rotate points by degrees around origin.
    
    Args:
    p (np.array): Points to rotate
    degrees (float): Rotation angle in degrees
    origin (tuple): Origin point for rotation
    
    Returns:
    np.array: Rotated points
    """
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def Dmat(P):
    """
    Calculate pairwise distance matrix.
    
    Args:
    P (np.array): Array of points
    
    Returns:
    np.array: Pairwise distance matrix
    """
    n = P.shape[0]
    D = np.zeros([n,n])
    for i in range(n):
        for j in range(i+1, n):
            D[i,j] = D[j,i] = sm.mae(P[i], P[j])
    return D

# Convert pairwise distance matrix to vector
inv_squareform = lambda a: a[np.nonzero(np.triu(a))]

def main(args):
    """
    Main function to stabilize the video.
    
    Args:
    args (argparse.Namespace): Command-line arguments
    """
    # Set default reference video if not provided
    if 'ref' not in args or args.ref is None:
        args.ref = args.input

    # Generate output filename if not provided
    if 'output' not in args or args.output is None:
        name, ext = ntpath.splitext(args.input)
        args.output = name + '_stabilized' + ext
        i = 1
        while os.path.exists(args.output):
            args.output = name + f'_stabilized_{i}' + ext
            i += 1
    
    # Read reference frames with CLAHE applied
    clahe = [8, 12]
    frames, fps = read_frames(args.ref, clahe=clahe)
    images = {}
    
    j = 0
    img1 = images[j] = frames[j]

    # Parameters for Lucas-Kanade optical flow
    win = 15
    lk_params = dict(winSize=(win, win),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Parameters for Shi-Tomasi Corner Detection
    minCorners = args.min_features
    maxCorners = minCorners + 50
    bs = 7
    lambda_min = 0.3

    # Find enough corners by reducing threshold
    while True:
        feature_params = dict(maxCorners=maxCorners,
                              qualityLevel=lambda_min,
                              minDistance=bs,
                              blockSize=bs)
        p0 = cv.goodFeaturesToTrack(img1, mask=None, **feature_params)
        if p0.shape[0] > minCorners:
            print(f'qualityLevel={lambda_min:0.4f} corners={p0.shape[0]}')
            break
        lambda_min *= 0.9
        
    # Track corners across frames
    F = len(frames)
    P = np.zeros([p0.shape[0], 2, F])
    P[:,:,0] = p0.squeeze()

    for j in range(1, len(frames)):
        img2 = images[j] = frames[j]

        # Calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)
        
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            z = st.squeeze()
            P = P[z==1]
            P[:,:,j] = good_new.squeeze()

        # Update previous frame and points
        img1 = img2.copy()
        p0 = good_new.reshape(-1, 1, 2)
    
    # Normalize optical flow tracks
    F = len(P)
    P = np.transpose(P, (0, 2, 1))

    # Shift all paths to start at (0,0)
    P0 = P - P[:,[0],:]

    # Rotation normalize - Rotate so P.mean() is level with P[0]
    Pm = P0.mean(1)
    A = np.rad2deg(np.arctan(Pm[:,1]/Pm[:,0]))
    Pr = np.array([rotate(p, -a) for p, a in zip(P0, A)])
    
    # Get pairwise distance matrix between tracks
    D0 = Dmat(P0)
    Dr = Dmat(Pr)
    D = D0 if D0.mean()<Dr.mean() else Dr
    d = inv_squareform(D)
    
    # Filter optical flow tracks to get best keypoint pairs
    X, Y = [], []
    for x in np.linspace(0.05, .95, 50):
        q = np.quantile(d, x)
        G = nx.Graph(D<q)
        mc = np.array(max(nx.find_cliques(G), key=len))
        y = D[mc[:,None],mc[None,:]].max()
        X.append(x)
        Y.append(y)
    
    # Find elbow point
    X, Y = np.array(X), np.array(Y)
    Xn, Yn = (X-X.min())/(X.max()-X.min()), (Y-Y.min())/(Y.max()-Y.min())
    dYn = np.gradient(Yn, Xn)

    L = 0.95  # arbitrary cutoff
    q = np.where(dYn > np.quantile(dYn, L))[0][0]
    x = X[q]
    
    # Get best keypoint pairs
    q = np.quantile(d, x)
    G = nx.Graph(D<q)
    idx = np.array(max(nx.find_cliques(G), key=len))
    idx.sort()
    K = P[idx]
    K0 = P0[idx]

    # Get best reference frame
    i = args.center
    if i < 0:
        S = K0.mean(0)
        i = np.sum((S-S.mean(0))**2, 1).argmin()
        print(f'center={i}')

    # Read input frames
    clahe = [] if args.clahe < 1 else [args.clahe]
    frames, fps = read_frames(args.input, clahe=clahe)
    img1 = frames[i]
    h, w = img1.shape

    # Transform frames
    k1 = K[:,[i],:]
    c = np.array([[0,0,1],[w,h,1],[0,h,1],[w,0,1]])
    C = []
    
    for j, img2 in enumerate(frames):
        if i == j:
            continue
        
        k2 = K[:,[j],:]
        if args.trans == 0:  # Affine
            T, Tmask = cv.estimateAffinePartial2D(k2, k1)
            img2T = cv.warpAffine(img2, T, (w, h))
        else:  # Homography
            T, Tmask = cv.findHomography(k2, k1, cv.RANSAC, 5.0)
            img2T = cv.warpPerspective(img2, T, (w, h))

        # Save transformed frame
        img2T = match_histograms(img2T, img2)
        frames[j] = img2T
        C.append(c @ T.T)

    # Crop frames to common region
    if args.crop:
        C = np.array(C)
        x0 = int(np.ceil(C[:,c[:,0]==0,0].max()))
        x1 = int(np.floor(C[:,c[:,0]>0,0].min()))
        y0 = int(np.ceil(C[:,c[:,1]==0,1].max()))
        y1 = int(np.floor(C[:,c[:,1]>0,1].min()))
        frames = [frame[y0:y1, x0:x1] for frame in frames]
        
    # Save stabilized video
    write_frames(frames, args.output, fps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stabilize video')
    parser.add_argument('--input', help='path of input video', type=str, required=True)
    parser.add_argument('--ref', help='path of reference video (for tracking)', type=str, default=None)
    parser.add_argument('--output', help='path of output video', type=str, default=None)
    parser.add_argument('--trans', help='transform type (0=Affine, 1=Homography)', type=int, default=0)
    parser.add_argument('--min_features', help='min Shi-Tomasi corners', type=int, default=60)
    parser.add_argument('--center', help='center frame (-1 triggers auto-search)', type=int, default=-1)
    parser.add_argument('--clahe', help='apply CLAHE to output', type=int, default=8)
    parser.add_argument('--crop', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    main(args)