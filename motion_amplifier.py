import torch
import pyramid
import temporal_filter as tf
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
import numpy as np

def magnify_motion(tensor, fps, low_freq, high_freq, alpha, levels, spatial_blur_sigma=0,
                   motion_threshold=0, adaptive_amplify=False, bilateral_filter=False,
                   color_stabilize=False, multiband=False):
    """
    使用增强控制和降噪功能放大视频张量中的运动。
    
    参数:
    -----------
    tensor : torch.Tensor
        形状为(T, C, H, W)的输入视频张量
    fps : float
        视频的每秒帧数
    low_freq : float
        带通滤波器的低截止频率(Hz)
    high_freq : float
        带通滤波器的高截止频率(Hz)
    alpha : float
        基本运动放大系数
    levels : int
        拉普拉斯金字塔的层数
    spatial_blur_sigma : float, optional
        运动信号高斯模糊的标准差(0表示禁用)
    motion_threshold : float, optional
        低于此阈值的运动不会被放大(0表示禁用)
    adaptive_amplify : bool, optional
        如果为True，应用层自适应放大(较低频率获得更强的放大)
    bilateral_filter : bool, optional
        如果为True，使用双边滤波而不是高斯模糊以更好地保留边缘
    color_stabilize : bool, optional
        如果为True，应用颜色稳定以减少闪烁
    multiband : bool, optional
        如果为True，将频率范围分为多个具有自适应放大的频段
    
    返回:
    --------
    torch.Tensor
        运动放大后的视频张量
    """
    # 如果启用，应用颜色稳定(在金字塔分解之前)
    if color_stabilize:
        tensor = _stabilize_colors(tensor)
    
    # 构建拉普拉斯金字塔
    lap_pyramid = pyramid.build_laplacian_pyramid(tensor, levels)
    
    # 如果启用，处理多个频率段
    if multiband:
        frequency_bands = [
            (low_freq, (low_freq + high_freq) / 2, alpha * 1.2),  # 较低频率 - 更强的放大
            ((low_freq + high_freq) / 2, high_freq, alpha * 0.8)   # 较高频率 - 更温和的放大
        ]
    else:
        frequency_bands = [(low_freq, high_freq, alpha)]
    
    # 应用时域滤波，放大，并原地添加回金字塔
    for i in range(levels):
        # 计算层特定的放大系数(用于自适应模式)
        level_alpha_factor = 1.0
        if adaptive_amplify:
            # 较低层(较大结构)获得更多放大
            # 较高层(更精细的细节)获得较少放大
            level_alpha_factor = 1.0 - 0.25 * (i / (levels - 1))
        
        for band_low, band_high, band_alpha in frequency_bands:
            # 为此频率段过滤当前层
            filtered_level = tf.butterworth_bandpass_filter(lap_pyramid[i], band_low, band_high, fps)
            
            # 如果启用，应用运动阈值
            if motion_threshold > 0:
                # 计算运动幅度
                motion_magnitude = torch.abs(filtered_level)
                # 创建运动高于阈值的掩码
                motion_mask = (motion_magnitude > motion_threshold).float()
                # 平滑掩码以避免突变
                kernel_size = 5
                motion_mask = F.avg_pool2d(
                    motion_mask.reshape(-1, 1, motion_mask.shape[2], motion_mask.shape[3]),
                    kernel_size, stride=1, padding=kernel_size//2
                ).reshape(filtered_level.shape)
                # 将掩码应用于过滤后的层
                filtered_level = filtered_level * motion_mask

            # 如果启用，应用空间滤波
            if spatial_blur_sigma > 0:
                if bilateral_filter:
                    # 边缘保留双边滤波(PyTorch中的近似)
                    filtered_level = _bilateral_filter_approximation(filtered_level, spatial_blur_sigma)
                else:
                    # 标准高斯模糊
                    kernel_size = int(2 * round(2.5 * spatial_blur_sigma) + 1)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    blur = GaussianBlur(kernel_size, sigma=spatial_blur_sigma)
                    # 重塑为(B*C, H, W)进行模糊处理
                    T, C, H, W = filtered_level.shape
                    filtered_level = blur(filtered_level.reshape(T * C, 1, H, W)).reshape(T, C, H, W)

            # 放大并原地添加回
            lap_pyramid[i] += filtered_level * (band_alpha * level_alpha_factor)
            
            # 清理以节省内存
            del filtered_level

    # 折叠金字塔
    result_tensor = pyramid.collapse_laplacian_pyramid(lap_pyramid)
    
    # 确保结果张量在有效范围[0, 1]内
    result_tensor = torch.clamp(result_tensor, 0, 1)
    
    return result_tensor

def _bilateral_filter_approximation(tensor, sigma_spatial, sigma_color=0.1):
    """
    使用高斯模糊和权重掩码的组合近似双边滤波。
    这比标准高斯模糊更好地保留边缘。
    
    参数:
    -----------
    tensor : torch.Tensor
        要过滤的输入张量
    sigma_spatial : float
        滤波器的空间sigma
    sigma_color : float
        滤波器的颜色sigma
        
    返回:
    --------
    torch.Tensor
        过滤后的张量
    """
    # 创建高斯模糊
    kernel_size = int(2 * round(2.5 * sigma_spatial) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    blur = GaussianBlur(kernel_size, sigma=sigma_spatial)
    
    # 处理每一帧
    result = []
    T, C, H, W = tensor.shape
    
    for t in range(T):
        frame = tensor[t]
        # 计算颜色差异权重
        blurred = blur(frame.unsqueeze(0))[0]
        diff = torch.abs(frame - blurred)
        weight = torch.exp(-(diff / sigma_color)**2)
        
        # 应用权重控制过滤强度
        filtered = blurred * weight + frame * (1 - weight)
        result.append(filtered)
    
    return torch.stack(result)

def _stabilize_colors(tensor):
    """
    稳定帧间颜色以减少闪烁。
    
    参数:
    -----------
    tensor : torch.Tensor
        形状为(T, C, H, W)的输入视频张量
        
    返回:
    --------
    torch.Tensor
        颜色稳定后的视频张量
    """
    T, C, H, W = tensor.shape
    result = tensor.clone()
    
    if T <= 1:
        return result
    
    # 计算每帧的平均颜色
    frame_means = torch.mean(tensor, dim=[2, 3])  # 形状: (T, C)
    
    # 计算总体平均颜色
    overall_mean = torch.mean(frame_means, dim=0)  # 形状: (C)
    
    # 应用颜色校正以稳定颜色
    for t in range(T):
        for c in range(C):
            # 调整颜色的比例因子
            scale = overall_mean[c] / frame_means[t, c] if frame_means[t, c] > 0.01 else 1.0
            # 应用温和的校正(部分调整)
            result[t, c] = tensor[t, c] * (0.5 + 0.5 * scale)
    
    return result

def suggest_parameters(fps, input_resolution=None, motion_type='general'):
    """
    根据视频属性建议运动放大的最佳参数。
    
    参数:
    -----------
    fps : float
        视频的每秒帧数
    input_resolution : tuple, optional
        输入视频的分辨率(宽度, 高度)
    motion_type : str, optional
        要放大的运动类型('general', 'pulse', 'breathing', 'subtle', 'large')
        
    返回:
    --------
    dict
        包含建议参数的字典
    """
    params = {}
    
    # 基于运动类型的基本频率
    if motion_type == 'pulse':
        params['freq_min'] = 0.8
        params['freq_max'] = 2.0
        params['alpha'] = 15.0
    elif motion_type == 'breathing':
        params['freq_min'] = 0.1
        params['freq_max'] = 0.5
        params['alpha'] = 10.0
    elif motion_type == 'subtle':
        params['freq_min'] = 0.5
        params['freq_max'] = 2.0
        params['alpha'] = 5.0
    elif motion_type == 'large':
        params['freq_min'] = 0.2
        params['freq_max'] = 4.0
        params['alpha'] = 20.0
    else:  # general
        params['freq_min'] = 0.4
        params['freq_max'] = 3.0
        params['alpha'] = 10.0
    
    # 根据fps调整参数
    if fps < 24:
        # 较低帧率视频需要调整频率段
        params['freq_max'] = min(params['freq_max'], fps / 8)
    elif fps > 60:
        # 较高帧率视频可以捕获更高频率
        params['freq_max'] = min(params['freq_max'] * 1.5, fps / 4)
    
    # 根据分辨率调整参数
    if input_resolution:
        width, height = input_resolution
        resolution = width * height
        
        # 较高分辨率视频可能受益于更多的金字塔层
        if resolution > 1920 * 1080:
            params['levels'] = 4
            params['blur'] = 0.8
        elif resolution > 1280 * 720:
            params['levels'] = 3
            params['blur'] = 0.5
        else:
            params['levels'] = 3
            params['blur'] = 0.3
    else:
        # 如果分辨率未知，则使用默认值
        params['levels'] = 3
        params['blur'] = 0.0
    
    # 处理参数
    if input_resolution and (width * height > 1920 * 1080):
        # 对于4K或更高视频
        params['chunk_size'] = 15
        params['overlap'] = 5
    else:
        params['chunk_size'] = 20
        params['overlap'] = 8
    
    return params
