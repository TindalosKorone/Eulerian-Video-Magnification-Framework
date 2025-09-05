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
    Magnifies motion in a video tensor with enhanced control and noise reduction.
    
    Parameters:
    -----------
    tensor : torch.Tensor
        Input video tensor of shape (T, C, H, W)
    fps : float
        Frames per second of the video
    low_freq : float
        Low cutoff frequency for bandpass filter (Hz)
    high_freq : float
        High cutoff frequency for bandpass filter (Hz)
    alpha : float
        Base motion amplification factor
    levels : int
        Number of levels in the Laplacian pyramid
    spatial_blur_sigma : float, optional
        Standard deviation for Gaussian blur of the motion signal (0 to disable)
    motion_threshold : float, optional
        Threshold below which motion is not amplified (0 to disable)
    adaptive_amplify : bool, optional
        If True, apply layer-adaptive amplification (stronger for lower frequencies)
    bilateral_filter : bool, optional
        If True, use bilateral filtering instead of Gaussian blur for better edge preservation
    color_stabilize : bool, optional
        If True, apply color stabilization to reduce flickering
    multiband : bool, optional
        If True, divide the frequency range into multiple bands with adaptive amplification
    
    Returns:
    --------
    torch.Tensor
        Motion magnified video tensor
    """
    # Apply color stabilization if enabled (before pyramid decomposition)
    if color_stabilize:
        tensor = _stabilize_colors(tensor)
    
    # Build Laplacian pyramid
    lap_pyramid = pyramid.build_laplacian_pyramid(tensor, levels)
    
    # Process multiple frequency bands if enabled
    if multiband:
        frequency_bands = [
            (low_freq, (low_freq + high_freq) / 2, alpha * 1.2),  # Lower frequencies - stronger amplification
            ((low_freq + high_freq) / 2, high_freq, alpha * 0.8)   # Higher frequencies - gentler amplification
        ]
    else:
        frequency_bands = [(low_freq, high_freq, alpha)]
    
    # Apply temporal filter, amplify, and add back to the pyramid in-place
    for i in range(levels):
        # Calculate layer-specific amplification factor (for adaptive mode)
        level_alpha_factor = 1.0
        if adaptive_amplify:
            # Lower levels (larger structures) get more amplification
            # Higher levels (finer details) get less amplification
            level_alpha_factor = 1.0 - 0.25 * (i / (levels - 1))
        
        for band_low, band_high, band_alpha in frequency_bands:
            # Filter the current level for this frequency band
            filtered_level = tf.butterworth_bandpass_filter(lap_pyramid[i], band_low, band_high, fps)
            
            # Apply motion threshold if enabled
            if motion_threshold > 0:
                # Calculate motion magnitude
                motion_magnitude = torch.abs(filtered_level)
                # Create mask where motion is above threshold
                motion_mask = (motion_magnitude > motion_threshold).float()
                # Smooth mask to avoid abrupt transitions
                kernel_size = 5
                motion_mask = F.avg_pool2d(
                    motion_mask.reshape(-1, 1, motion_mask.shape[2], motion_mask.shape[3]),
                    kernel_size, stride=1, padding=kernel_size//2
                ).reshape(filtered_level.shape)
                # Apply mask to filtered level
                filtered_level = filtered_level * motion_mask

            # Apply spatial filtering if enabled
            if spatial_blur_sigma > 0:
                if bilateral_filter:
                    # Edge-preserving bilateral filter (approximation in PyTorch)
                    filtered_level = _bilateral_filter_approximation(filtered_level, spatial_blur_sigma)
                else:
                    # Standard Gaussian blur
                    kernel_size = int(2 * round(2.5 * spatial_blur_sigma) + 1)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    blur = GaussianBlur(kernel_size, sigma=spatial_blur_sigma)
                    # Reshape to (B*C, H, W) for blurring
                    T, C, H, W = filtered_level.shape
                    filtered_level = blur(filtered_level.reshape(T * C, 1, H, W)).reshape(T, C, H, W)

            # Amplify and add back in-place
            lap_pyramid[i] += filtered_level * (band_alpha * level_alpha_factor)
            
            # Clean up to save memory
            del filtered_level

    # Collapse the pyramid
    result_tensor = pyramid.collapse_laplacian_pyramid(lap_pyramid)
    
    # Ensure resulting tensor is within valid range [0, 1]
    result_tensor = torch.clamp(result_tensor, 0, 1)
    
    return result_tensor

def _bilateral_filter_approximation(tensor, sigma_spatial, sigma_color=0.1):
    """
    Approximate bilateral filtering using a combination of Gaussian blur and weight masks.
    This preserves edges better than standard Gaussian blur.
    
    Parameters:
    -----------
    tensor : torch.Tensor
        Input tensor to filter
    sigma_spatial : float
        Spatial sigma for the filter
    sigma_color : float
        Color sigma for the filter
        
    Returns:
    --------
    torch.Tensor
        Filtered tensor
    """
    # Create Gaussian blur
    kernel_size = int(2 * round(2.5 * sigma_spatial) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    blur = GaussianBlur(kernel_size, sigma=sigma_spatial)
    
    # Process each frame
    result = []
    T, C, H, W = tensor.shape
    
    for t in range(T):
        frame = tensor[t]
        # Compute color difference weight
        blurred = blur(frame.unsqueeze(0))[0]
        diff = torch.abs(frame - blurred)
        weight = torch.exp(-(diff / sigma_color)**2)
        
        # Apply weight to control filtering strength
        filtered = blurred * weight + frame * (1 - weight)
        result.append(filtered)
    
    return torch.stack(result)

def _stabilize_colors(tensor):
    """
    Stabilize colors across frames to reduce flickering.
    
    Parameters:
    -----------
    tensor : torch.Tensor
        Input video tensor of shape (T, C, H, W)
        
    Returns:
    --------
    torch.Tensor
        Color-stabilized video tensor
    """
    T, C, H, W = tensor.shape
    result = tensor.clone()
    
    if T <= 1:
        return result
    
    # Calculate mean color for each frame
    frame_means = torch.mean(tensor, dim=[2, 3])  # Shape: (T, C)
    
    # Calculate overall mean color
    overall_mean = torch.mean(frame_means, dim=0)  # Shape: (C)
    
    # Apply color correction to stabilize colors
    for t in range(T):
        for c in range(C):
            # Scale factor to adjust colors
            scale = overall_mean[c] / frame_means[t, c] if frame_means[t, c] > 0.01 else 1.0
            # Apply gentle correction (partial adjustment)
            result[t, c] = tensor[t, c] * (0.5 + 0.5 * scale)
    
    return result

def suggest_parameters(fps, input_resolution=None, motion_type='general'):
    """
    Suggest optimal parameters for motion magnification based on video properties.
    
    Parameters:
    -----------
    fps : float
        Frames per second of the video
    input_resolution : tuple, optional
        Resolution of the input video (width, height)
    motion_type : str, optional
        Type of motion to amplify ('general', 'pulse', 'breathing', 'subtle', 'large')
        
    Returns:
    --------
    dict
        Dictionary containing suggested parameters
    """
    params = {}
    
    # Base frequencies based on motion type
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
    
    # Adjust parameters based on fps
    if fps < 24:
        # Lower frame rate videos need adjusted frequency bands
        params['freq_max'] = min(params['freq_max'], fps / 8)
    elif fps > 60:
        # Higher frame rate videos can capture higher frequencies
        params['freq_max'] = min(params['freq_max'] * 1.5, fps / 4)
    
    # Adjust parameters based on resolution
    if input_resolution:
        width, height = input_resolution
        resolution = width * height
        
        # Higher resolution videos might benefit from more pyramid levels
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
        # Default values if resolution is unknown
        params['levels'] = 3
        params['blur'] = 0.0
    
    # Processing parameters
    if input_resolution and (width * height > 1920 * 1080):
        # For 4K or higher videos
        params['chunk_size'] = 15
        params['overlap'] = 5
    else:
        params['chunk_size'] = 20
        params['overlap'] = 8
    
    return params
