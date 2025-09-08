import torch
import torch.nn.functional as F

def build_laplacian_pyramid(tensor, levels):
    """为帧张量构建拉普拉斯金字塔。"""
    pyramid = []
    current = tensor
    for _ in range(levels):
        down = F.interpolate(current, scale_factor=0.5, mode='bilinear', align_corners=False)
        up = F.interpolate(down, size=current.shape[2:], mode='bilinear', align_corners=False)
        lap = current - up
        pyramid.append(lap)
        current = down
    pyramid.append(current)
    return pyramid

def collapse_laplacian_pyramid(pyramid):
    """折叠拉普拉斯金字塔以重建原始张量。"""
    current = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        up = F.interpolate(current, size=pyramid[i].shape[2:], mode='bilinear', align_corners=False)
        current = up + pyramid[i]
    return current
