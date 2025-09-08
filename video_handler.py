import cv2
import torch
import numpy as np

def read_video(path):
    """读取视频并返回其帧、fps、宽度和高度。"""
    vidcap = cv2.VideoCapture(path)
    if not vidcap.isOpened():
        raise IOError(f"无法打开位于 {path} 的视频。")
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    while True:
        success, image = vidcap.read()
        if not success:
            break
        frames.append(image)
    vidcap.release()
    
    return frames, fps, width, height

def write_video(frames, path, fps, width, height):
    """将帧列表写入视频文件。"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()

def frames_to_tensor(frames, device):
    """将帧列表转换为PyTorch张量。"""
    images = np.stack(frames)
    images = images.astype(np.float32) / 255.0
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).to(device)
    return tensor

def tensor_to_frames(tensor):
    """将PyTorch张量转换为帧列表。"""
    tensor = tensor.permute(0, 2, 3, 1).cpu().numpy()
    tensor = (tensor * 255.0).clip(0, 255).astype(np.uint8)
    frames = [frame for frame in tensor]
    return frames
