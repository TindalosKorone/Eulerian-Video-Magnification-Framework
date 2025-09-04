import torch
import pyramid
import temporal_filter as tf
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

def magnify_motion(tensor, fps, low_freq, high_freq, alpha, levels, spatial_blur_sigma=0):
    """Magnifies motion in a video tensor."""
    # Build Laplacian pyramid
    lap_pyramid = pyramid.build_laplacian_pyramid(tensor, levels)
    
    # Apply temporal filter, amplify, and add back to the pyramid in-place
    for i in range(levels):
        # Filter the current level
        filtered_level = tf.butterworth_bandpass_filter(lap_pyramid[i], low_freq, high_freq, fps)

        # Apply spatial blur if sigma is greater than 0
        if spatial_blur_sigma > 0:
            kernel_size = int(2 * round(2.5 * spatial_blur_sigma) + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            blur = GaussianBlur(kernel_size, sigma=spatial_blur_sigma)
            # Reshape to (B*C, H, W) for blurring
            T, C, H, W = filtered_level.shape
            filtered_level = blur(filtered_level.reshape(T * C, 1, H, W)).reshape(T, C, H, W)

        # Amplify and add back in-place
        lap_pyramid[i] += filtered_level * alpha
        
        # Clean up to save memory
        del filtered_level

    # Collapse the pyramid
    result_tensor = pyramid.collapse_laplacian_pyramid(lap_pyramid)
    
    return result_tensor
