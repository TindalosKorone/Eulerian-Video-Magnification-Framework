import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SteerablePyramid(nn.Module):
    """
    A batch-aware PyTorch implementation of a Complex Steerable Pyramid.
    """
    def __init__(self, height, width, order=3, num_scales='auto'):
        super(SteerablePyramid, self).__init__()
        self.M, self.N = height, width
        self.order = order
        self.num_orientations = order + 1
        
        if num_scales == 'auto':
            self.num_scales = int(np.floor(np.log2(min(self.M, self.N))) - 2)
        else:
            self.num_scales = num_scales

        self.build_filters()

    def build_filters(self):
        # Create frequency domain coordinates
        x, y = np.meshgrid(np.linspace(-self.N/2, self.N/2-1, self.N), np.linspace(-self.M/2, self.M/2-1, self.M))
        radius = np.fft.ifftshift(np.sqrt(x**2 + y**2))
        angle = np.fft.ifftshift(np.arctan2(y, x))
        
        # Highpass filter
        self.register_buffer('highpass_filter', torch.from_numpy(self._highpass_mask(radius)).float())

        # Lowpass filter (for the residual)
        self.register_buffer('lowpass_filter', torch.from_numpy(self._lowpass_mask(radius)).float())

        # Angular (orientation) filters
        angle_filters = torch.stack([torch.from_numpy(self._angle_mask(angle, k)).float() for k in range(self.num_orientations)])
        self.register_buffer('angle_filters', angle_filters)

        # Radial (scale) filters
        scale_filters = torch.stack([torch.from_numpy(self._scale_mask(radius, j)).float() for j in range(self.num_scales)])
        self.register_buffer('scale_filters', scale_filters)

    def _highpass_mask(self, radius):
        return (radius >= min(self.M, self.N) / 4).astype(float)

    def _lowpass_mask(self, radius):
        return (radius < min(self.M, self.N) / (2**(self.num_scales+2))).astype(float)

    def _angle_mask(self, angle, k):
        angle_shift = angle - k * np.pi / self.num_orientations
        mask = (np.cos(angle_shift)**self.order) * (np.abs(angle_shift) < np.pi/2).astype(float)
        return mask

    def _scale_mask(self, radius, j):
        log_rad = np.log2(radius + 1e-12) # Add epsilon to avoid log(0)
        center = np.log2(min(self.M, self.N)) - (j + 2)
        mask = np.cos(np.pi * (log_rad - center)) * (np.abs(log_rad - center) < 0.5).astype(float)
        return mask

    def forward(self, x):
        # Expects x to be a 4D tensor (N, C, H, W)
        im_fft = torch.fft.fft2(x, dim=(-2, -1))

        # Highpass residual
        highpass_res = torch.fft.ifft2(im_fft * self.highpass_filter, dim=(-2, -1))

        # Bandpass components
        pyramid = {'highpass': highpass_res, 'bands': []}
        for j in range(self.num_scales):
            bandpass_level = []
            for k in range(self.num_orientations):
                band_filter = self.angle_filters[k] * self.scale_filters[j]
                band_fft = im_fft * band_filter
                band = torch.fft.ifft2(band_fft, dim=(-2, -1))
                # Downsample the spatial representation by interpolating real and imaginary parts separately
                downsample_factor = 2**(j+1)
                band_real_down = F.interpolate(band.real, scale_factor=1/downsample_factor, mode='bilinear', align_corners=False)
                band_imag_down = F.interpolate(band.imag, scale_factor=1/downsample_factor, mode='bilinear', align_corners=False)
                bandpass_level.append(band_real_down + 1j * band_imag_down)
            pyramid['bands'].append(torch.stack(bandpass_level, dim=2)) # Stack along new orientation dim

        # Lowpass residual
        lowpass_res = torch.fft.ifft2(im_fft * self.lowpass_filter, dim=(-2, -1))
        downsample_factor = 2**self.num_scales
        lowpass_real_down = F.interpolate(lowpass_res.real, scale_factor=1/downsample_factor, mode='bilinear', align_corners=False)
        lowpass_imag_down = F.interpolate(lowpass_res.imag, scale_factor=1/downsample_factor, mode='bilinear', align_corners=False)
        pyramid['lowpass'] = lowpass_real_down + 1j * lowpass_imag_down
        
        return pyramid

    def reconstruct(self, pyramid_coeffs):
        # Lowpass component
        lowpass_res = pyramid_coeffs['lowpass']
        target_size = (self.M, self.N)
        lowpass_real_up = F.interpolate(lowpass_res.real, size=target_size, mode='bilinear', align_corners=False)
        lowpass_imag_up = F.interpolate(lowpass_res.imag, size=target_size, mode='bilinear', align_corners=False)
        lowpass_upsampled = lowpass_real_up + 1j * lowpass_imag_up
        recon_fft = torch.fft.fft2(lowpass_upsampled, dim=(-2, -1)) * self.lowpass_filter

        # Bandpass components
        for j in range(self.num_scales):
            bandpass_level = pyramid_coeffs['bands'][j]
            for k in range(self.num_orientations):
                band = bandpass_level[:, :, k] # Select orientation
                band_real_up = F.interpolate(band.real, size=target_size, mode='bilinear', align_corners=False)
                band_imag_up = F.interpolate(band.imag, size=target_size, mode='bilinear', align_corners=False)
                band_upsampled = band_real_up + 1j * band_imag_up
                band_filter = self.angle_filters[k] * self.scale_filters[j]
                recon_fft += torch.fft.fft2(band_upsampled, dim=(-2, -1)) * band_filter

        # Highpass component
        highpass_res = pyramid_coeffs['highpass']
        recon_fft += torch.fft.fft2(highpass_res, dim=(-2, -1)) * self.highpass_filter

        recon = torch.fft.ifft2(recon_fft, dim=(-2, -1))
        return recon.real
