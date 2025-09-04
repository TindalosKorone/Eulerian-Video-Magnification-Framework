import cv2
import torch
import numpy as np

def read_video(path):
    """Reads a video and returns its frames, fps, width, and height."""
    vidcap = cv2.VideoCapture(path)
    if not vidcap.isOpened():
        raise IOError(f"Video at {path} could not be opened.")
    
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
    """Writes a list of frames to a video file."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()

def frames_to_tensor(frames, device):
    """Converts a list of frames to a PyTorch tensor."""
    images = np.stack(frames)
    images = images.astype(np.float32) / 255.0
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).to(device)
    return tensor

def tensor_to_frames(tensor):
    """Converts a PyTorch tensor to a list of frames."""
    tensor = tensor.permute(0, 2, 3, 1).cpu().numpy()
    tensor = (tensor * 255.0).clip(0, 255).astype(np.uint8)
    frames = [frame for frame in tensor]
    return frames
