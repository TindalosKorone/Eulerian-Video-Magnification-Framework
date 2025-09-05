import numpy as np
import cv2
from tqdm import tqdm

def stabilize_video(input_path, output_path, smoothing_radius=None, smoothing_strength=0.95):
    """
    Stabilizes a video by removing camera shake.
    
    Parameters:
    -----------
    input_path : str
        Path to the input video file
    output_path : str
        Path to save the stabilized video
    smoothing_radius : int, optional
        Radius (in frames) for trajectory smoothing.
        If None, will use the video's fps as a reasonable default (1 second window)
    smoothing_strength : float, optional
        Strength of the smoothing effect (0.0-1.0).
        Higher values result in more aggressive stabilization.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Video at {input_path} could not be opened.")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    transforms = np.zeros((n_frames-1, 3), np.float32)

    for i in tqdm(range(n_frames-2), desc="Stabilizing video (pass 1)"):
        success, curr = cap.read()
        if not success:
            break

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        assert prev_pts.shape == curr_pts.shape

        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        
        dx = m[0,2]
        dy = m[1,2]
        da = np.arctan2(m[1,0], m[0,0])
        
        transforms[i] = [dx, dy, da]
        
        prev_gray = curr_gray

    trajectory = np.cumsum(transforms, axis=0)
    
    # Set default smoothing radius if not provided
    if smoothing_radius is None:
        smoothing_radius = int(fps) # Default: smooth over a 1-second window
    
    # Apply smoothing
    smoothed_trajectory = np.zeros_like(trajectory)
    for i in range(trajectory.shape[0]):
        start = max(0, i - smoothing_radius)
        end = min(trajectory.shape[0], i + smoothing_radius)
        smoothed_trajectory[i] = np.mean(trajectory[start:end], axis=0)

    # Apply smoothing strength factor
    smoothed_trajectory = trajectory + smoothing_strength * (smoothed_trajectory - trajectory)
    
    # Calculate stabilization transforms
    transforms_smooth = transforms + smoothed_trajectory - trajectory

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    for i in tqdm(range(n_frames-1), desc="Stabilizing video (pass 2)"):
        success, frame = cap.read()
        if not success:
            break

        dx, dy, da = transforms_smooth[i]

        m = np.zeros((2,3), np.float32)
        m[0,0] = np.cos(da)
        m[0,1] = -np.sin(da)
        m[1,0] = np.sin(da)
        m[1,1] = np.cos(da)
        m[0,2] = dx
        m[1,2] = dy

        frame_stabilized = cv2.warpAffine(frame, m, (width,height))
        out.write(frame_stabilized)

    out.release()
    cap.release()
