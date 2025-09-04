import torch
import numpy as np
from steerable_pyramid import SteerablePyramid
import temporal_filter as tf

def magnify_phase(tensor, fps, low_freq, high_freq, alpha):
    """
    Magnifies motion in a video tensor using the phase-based method with batch processing.
    """
    # Assuming tensor is (T, C, H, W)
    T, C, H, W = tensor.shape
    
    # Initialize pyramid for the given dimensions
    pyramid_builder = SteerablePyramid(H, W).to(tensor.device)
    
    final_result_channels = []
    for c in range(C): # Process each color channel independently
        channel_tensor = tensor[:, c:c+1, :, :]
        
        # 1. Decompose the entire chunk at once
        pyramid_coeffs = pyramid_builder.forward(channel_tensor)

        # 2. Process each level and orientation in a vectorized way
        for j in range(pyramid_builder.num_scales):
            # Get the complex coefficients for this band over time (T, 1, K, H_level, W_level)
            coeffs = pyramid_coeffs['bands'][j]
            
            for k in range(pyramid_builder.num_orientations):
                band_coeffs = coeffs[:, :, k] # Shape: (T, 1, H_level, W_level)

                # Get amplitude and phase
                amplitude = torch.abs(band_coeffs)
                phase = torch.angle(band_coeffs)

                # GPU-friendly phase unwrap
                phase_diff = torch.diff(phase, dim=0)
                phase_diff_wrapped = torch.remainder(phase_diff + np.pi, 2 * np.pi) - np.pi
                phase_unwrapped = torch.cumsum(torch.cat([phase[0:1], phase_diff_wrapped], dim=0), dim=0)
                
                # Temporal filtering
                filtered_phase = tf.butterworth_bandpass_filter(phase_unwrapped, low_freq, high_freq, fps)

                # Amplify and add back
                amplified_phase = phase + filtered_phase * alpha

                # Create new complex coefficients
                new_coeffs = amplitude * torch.exp(1j * amplified_phase)

                # Put them back into the pyramid structure
                pyramid_coeffs['bands'][j][:, :, k] = new_coeffs

        # 3. Reconstruct the entire chunk at once
        reconstructed_channel = pyramid_builder.reconstruct(pyramid_coeffs)
        final_result_channels.append(reconstructed_channel)

    result_tensor = torch.cat(final_result_channels, dim=1) # Combine color channels
    
    return result_tensor
