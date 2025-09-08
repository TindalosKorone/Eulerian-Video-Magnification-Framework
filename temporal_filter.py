import torch
from torch.fft import fft, ifft

def butterworth_bandpass_filter(tensor, low_freq, high_freq, fps, order=5):
    """沿时间维度对张量应用巴特沃斯带通滤波器。"""
    T, C, H, W = tensor.shape
    freqs = torch.fft.fftfreq(T, d=1.0/fps, device=tensor.device)
    
    # 创建滤波器
    low_mask = (1 / (1 + (freqs / high_freq)**(2 * order)))
    high_mask = (1 - 1 / (1 + (freqs / low_freq)**(2 * order)))
    mask = low_mask * high_mask
    
    # 应用滤波器
    fft_tensor = fft(tensor, dim=0)
    fft_tensor *= mask.view(-1, 1, 1, 1)
    filtered_tensor = ifft(fft_tensor, dim=0).real
    
    return filtered_tensor
