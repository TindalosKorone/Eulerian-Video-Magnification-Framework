import numpy as np
import cv2
from tqdm import tqdm

def stabilize_video(input_path, output_path, smoothing_radius=None, smoothing_strength=0.95):
    """
    通过消除相机抖动来稳定视频。
    
    参数:
    -----------
    input_path : str
        输入视频文件的路径
    output_path : str
        保存稳定视频的路径
    smoothing_radius : int, optional
        轨迹平滑的半径（以帧为单位）。
        如果为None，将使用视频的fps作为合理的默认值（1秒窗口）
    smoothing_strength : float, optional
        平滑效果的强度（0.0-1.0）。
        较高的值会产生更激进的稳定效果。
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"无法打开位于 {input_path} 的视频。")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    transforms = np.zeros((n_frames-1, 3), np.float32)

    for i in tqdm(range(n_frames-2), desc="稳定视频（第1阶段）"):
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
    
    # 如果未提供平滑半径，则设置默认值
    if smoothing_radius is None:
        smoothing_radius = int(fps) # 默认：在1秒窗口上平滑
    
    # 应用平滑
    smoothed_trajectory = np.zeros_like(trajectory)
    for i in range(trajectory.shape[0]):
        start = max(0, i - smoothing_radius)
        end = min(trajectory.shape[0], i + smoothing_radius)
        smoothed_trajectory[i] = np.mean(trajectory[start:end], axis=0)

    # 应用平滑强度因子
    smoothed_trajectory = trajectory + smoothing_strength * (smoothed_trajectory - trajectory)
    
    # 计算稳定变换
    transforms_smooth = transforms + smoothed_trajectory - trajectory

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    for i in tqdm(range(n_frames-1), desc="稳定视频（第2阶段）"):
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
