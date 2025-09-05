import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import json
import hashlib

def analyze_video_frequencies(video_path, output_dir=None, sampling_rate=0.5, max_points=200, 
                             downsample_factor=1, max_freq_bands=20, generate_visualizations=True,
                             max_processing_time=None):
    """
    Analyze motion frequencies in a video and generate visualization of frequency hotspots.
    
    Parameters:
    -----------
    video_path : str
        Path to the input video file
    output_dir : str, optional
        Directory to save analysis results. If None, use same directory as video
    sampling_rate : float, optional
        Fraction of frames to analyze (0.0-1.0), lower values speed up analysis
    max_points : int, optional
        Maximum number of feature points to track per frame
        
    Returns:
    --------
    dict
        Dictionary containing analysis results and paths to generated visualizations
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(video_path))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract base name for output files
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Video at {video_path} could not be opened.")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate sampling based on rate
    sample_step = max(1, int(1/sampling_rate))
    sampled_frames = range(0, n_frames, sample_step)
    analysis_fps = fps / sample_step
    
    print(f"Analyzing video: {video_path}")
    print(f"Original FPS: {fps}, Analysis FPS: {analysis_fps}")
    print(f"Sampling 1 out of every {sample_step} frames ({len(sampled_frames)} frames total)")
    
    # Storage for motion data
    motion_data = []
    
    # Create grid of points to track (less dense than pixels)
    grid_step = max(1, int(min(width, height) / 30))
    grid_x = np.arange(0, width, grid_step)
    grid_y = np.arange(0, height, grid_step)
    grid_points = np.array([[x, y] for y in grid_y for x in grid_x], dtype=np.float32)
    
    # Initialize spatial heatmap (accumulate motion magnitude for each pixel)
    spatial_heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Read first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_frame = cap.read()
    if not ret:
        raise IOError("Failed to read first frame.")
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Track motion across frames
    frame_idx = 0
    tracked_points = []
    point_trajectories = {}  # Store motion time series for each point
    
    for frame_num in tqdm(sampled_frames[1:], desc="Analyzing motion"):
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Every 10 frames, refresh the set of points to track
        if frame_idx % 10 == 0 or len(tracked_points) < max_points/2:
            # Find good features to track
            new_points = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=max_points, qualityLevel=0.01, 
                minDistance=min(width, height) // 30, blockSize=7
            )
            
            if new_points is not None:
                if len(tracked_points) == 0:
                    tracked_points = new_points
                else:
                    tracked_points = np.vstack((tracked_points, new_points))
                    # Remove duplicate points
                    tracked_points = np.unique(tracked_points.reshape(-1, 2), axis=0).reshape(-1, 1, 2)
        
        if len(tracked_points) == 0:
            prev_gray = curr_gray
            continue
        
        # Calculate optical flow
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, tracked_points, None
        )
        
        # Filter out points that couldn't be tracked
        good_points = tracked_points[status == 1]
        good_next_points = next_points[status == 1]
        
        # Calculate motion vectors
        motion_vectors = good_next_points - good_points
        
        # Update point trajectories
        for i, (pt, motion) in enumerate(zip(good_points, motion_vectors)):
            # Handle different possible shapes of point arrays
            if pt.ndim == 1:  # pt is [x, y]
                x, y = int(pt[0]), int(pt[1])
            elif pt.ndim == 2:  # pt is [[x, y]]
                x, y = int(pt[0][0]), int(pt[0][1])
            elif pt.ndim == 3:  # pt is [[[x, y]]]
                x, y = int(pt[0, 0]), int(pt[0, 1])
            else:
                print(f"Warning: Unexpected point shape: {pt.shape}")
                continue
                
            pt_id = f"{x},{y}"
            if pt_id not in point_trajectories:
                point_trajectories[pt_id] = []
            
            # Handle different possible shapes of motion vectors
            if motion.ndim == 1:  # motion is [dx, dy]
                magnitude = np.sqrt(motion[0]**2 + motion[1]**2)
            elif motion.ndim == 2:  # motion is [[dx, dy]]
                magnitude = np.sqrt(motion[0][0]**2 + motion[0][1]**2)
            elif motion.ndim == 3:  # motion is [[[dx, dy]]]
                magnitude = np.sqrt(motion[0, 0]**2 + motion[0, 1]**2)
            else:
                print(f"Warning: Unexpected motion shape: {motion.shape}")
                continue
                
            point_trajectories[pt_id].append(magnitude)
            
            # Update spatial heatmap
            if 0 <= x < width and 0 <= y < height:
                spatial_heatmap[y, x] += magnitude
        
        # Update for next iteration
        prev_gray = curr_gray
        tracked_points = good_next_points.reshape(-1, 1, 2)
    
    cap.release()
    
    # Normalize spatial heatmap
    if np.max(spatial_heatmap) > 0:
        spatial_heatmap = spatial_heatmap / np.max(spatial_heatmap)
    
    # Smooth the spatial heatmap for better visualization
    spatial_heatmap = cv2.GaussianBlur(spatial_heatmap, (15, 15), 0)
    
    # Prepare data for frequency analysis
    # Find trajectories with enough data points
    min_traj_length = len(sampled_frames) // 2
    valid_trajectories = {pt_id: traj for pt_id, traj in point_trajectories.items() 
                         if len(traj) >= min_traj_length}
    
    print(f"Analyzing {len(valid_trajectories)} motion trajectories")
    
    # Define fixed frequency bands to limit memory usage
    max_freq = 10.0  # Maximum frequency to analyze (Hz)
    if max_freq_bands > 0:
        # Create evenly spaced frequency bands
        freq_band_step = max_freq / max_freq_bands
        predefined_freq_bands = np.arange(freq_band_step, max_freq + freq_band_step, freq_band_step)
    else:
        # Default: 0.5 Hz steps up to 10 Hz
        predefined_freq_bands = np.arange(0.5, 10.5, 0.5)
    
    # Initialize frequency maps with fixed bands
    frequency_maps = {}
    
    # If downsample_factor > 1, use smaller heatmaps to save memory
    map_height = height // downsample_factor
    map_width = width // downsample_factor
    
    for freq in predefined_freq_bands:
        frequency_maps[freq] = np.zeros((map_height, map_width), dtype=np.float32)
    
    # Perform frequency analysis on trajectories
    frequency_data = []
    
    print(f"Processing {len(valid_trajectories)} motion trajectories...")
    
    # Add a progress bar for frequency analysis phase
    for pt_id, trajectory in tqdm(valid_trajectories.items(), desc="Analyzing frequencies"):
        # Pad trajectory if needed
        if len(trajectory) < len(sampled_frames)-1:
            trajectory = trajectory + [0] * (len(sampled_frames)-1 - len(trajectory))
        
        # Apply window to reduce spectral leakage
        windowed_traj = trajectory * signal.windows.hann(len(trajectory))
        
        # Compute FFT
        fft_result = np.abs(np.fft.rfft(windowed_traj))
        freqs = np.fft.rfftfreq(len(trajectory), d=1/analysis_fps)
        
        # Store results
        frequency_data.append((freqs, fft_result, pt_id))
        
        # Contribute to frequency maps - optimized version
        x, y = map(int, pt_id.split(','))
        
        # Apply downsampling to coordinates
        map_x, map_y = x // downsample_factor, y // downsample_factor
        
        if 0 <= map_x < map_width and 0 <= map_y < map_height:
            # Find closest predefined frequency band for each frequency
            for i, freq in enumerate(freqs):
                if freq <= 0 or freq > max_freq:  # Ignore DC component and frequencies above max
                    continue
                
                # Find the closest predefined frequency band
                closest_band = min(predefined_freq_bands, key=lambda band: abs(band - freq))
                
                # Update the frequency map
                frequency_maps[closest_band][map_y, map_x] += fft_result[i]
    
    # Normalize and smooth frequency maps
    print("Normalizing and smoothing frequency maps...")
    for freq, freq_map in frequency_maps.items():
        if np.max(freq_map) > 0:
            frequency_maps[freq] = freq_map / np.max(freq_map)
            
        # Adjust the kernel size for smaller maps if downsampled
        kernel_size = max(3, 15 // downsample_factor)
        if kernel_size % 2 == 0:  # Ensure kernel size is odd
            kernel_size += 1
            
        # Smooth the map
        frequency_maps[freq] = cv2.GaussianBlur(frequency_maps[freq], (kernel_size, kernel_size), 0)
    
    # Generate combined frequency spectrum
    all_freqs = []
    all_amplitudes = []
    for freqs, fft_result, _ in frequency_data:
        all_freqs.extend(freqs)
        all_amplitudes.extend(fft_result)
    
    # Group by frequency and average
    unique_freqs = np.unique(all_freqs)
    avg_spectrum = np.zeros_like(unique_freqs)
    for i, freq in enumerate(unique_freqs):
        matching_indices = [j for j, f in enumerate(all_freqs) if f == freq]
        avg_spectrum[i] = np.mean([all_amplitudes[j] for j in matching_indices])
    
    # Normalize spectrum
    if np.max(avg_spectrum) > 0:
        avg_spectrum = avg_spectrum / np.max(avg_spectrum)
    
    # Skip visualization if requested
    if not generate_visualizations:
        # Create minimal visualization set
        visualization_paths = {
            'frequency_spectrum': None,
            'spatial_heatmap': None,
            'dominant_frequency_map': None
        }
        print("Skipping visualization generation")
    else:
        print("Generating visualizations...")
        visualization_paths = {}
    
    # 1. Frequency spectrum
    plt.figure(figsize=(12, 6))
    plt.plot(unique_freqs, avg_spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Relative Amplitude')
    plt.title('Frequency Spectrum of Motion')
    plt.grid(True)
    plt.xlim(0, min(10, np.max(unique_freqs)))  # Limit to 10 Hz or max freq
    
    # Add vertical lines at 0.5 Hz intervals
    for f in np.arange(0.5, min(10, np.max(unique_freqs)), 0.5):
        plt.axvline(x=f, color='r', linestyle='--', alpha=0.3)
    
    spectrum_path = os.path.join(output_dir, f"{base_name}_frequency_spectrum.png")
    plt.savefig(spectrum_path, dpi=150)
    plt.close()
    visualization_paths['frequency_spectrum'] = spectrum_path
    
    # 2. Spatial heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(spatial_heatmap, cmap='hot')
    plt.colorbar(label='Relative Motion Magnitude')
    plt.title('Spatial Motion Heatmap')
    plt.axis('off')
    
    heatmap_path = os.path.join(output_dir, f"{base_name}_spatial_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    visualization_paths['spatial_heatmap'] = heatmap_path
    
    # 3. Frequency-space heatmaps (for key frequencies)
    # Find top frequency bands by average energy
    freq_energies = [(freq, np.mean(freq_map)) for freq, freq_map in frequency_maps.items()]
    freq_energies.sort(key=lambda x: x[1], reverse=True)
    
    # Save only the top frequency maps to reduce processing time
    top_freq_maps = {}
    max_freq_maps = min(5, len(freq_energies))  # Save at most 5
    for freq, energy in freq_energies[:max_freq_maps]:
        plt.figure(figsize=(12, 8))
        plt.imshow(frequency_maps[freq], cmap='hot')
        plt.colorbar(label=f'Relative Motion at {freq:.1f} Hz')
        plt.title(f'Motion Heatmap at {freq:.1f} Hz')
        plt.axis('off')
        
        freq_map_path = os.path.join(output_dir, f"{base_name}_freq_{freq:.1f}Hz_heatmap.png")
        plt.savefig(freq_map_path, dpi=150)
        plt.close()
        top_freq_maps[str(freq)] = freq_map_path
    
    visualization_paths['frequency_maps'] = top_freq_maps
    
    # Create a combined frequency-motion heatmap
    # Weight each pixel by its frequency content
    dominant_freq_map = np.zeros((map_height, map_width), dtype=np.float32)
    max_energy_map = np.zeros((map_height, map_width), dtype=np.float32)
    
    for freq, freq_map in frequency_maps.items():
        # Update dominant frequency where this frequency has higher energy
        mask = freq_map > max_energy_map
        dominant_freq_map[mask] = freq
        max_energy_map[mask] = freq_map[mask]
    
    # Scale to a colormap range (0-1 for frequencies up to 5Hz)
    norm_freq_map = np.clip(dominant_freq_map / 5.0, 0, 1)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(norm_freq_map, cmap='jet')
    plt.colorbar(label='Dominant Frequency (Hz)')
    plt.title('Dominant Frequency Map')
    plt.axis('off')
    
    dom_freq_path = os.path.join(output_dir, f"{base_name}_dominant_freq_map.png")
    plt.savefig(dom_freq_path, dpi=150)
    plt.close()
    visualization_paths['dominant_frequency_map'] = dom_freq_path
    
    print("Generating suggested frequency ranges...")
    # Generate suggested frequency ranges based on the spectrum
    # Find peaks in the spectrum with less strict criteria if no major peaks found
    peaks, _ = signal.find_peaks(avg_spectrum, height=0.2, distance=int(0.5 / (unique_freqs[1] - unique_freqs[0])))
    if len(peaks) == 0:
        # Try again with lower height threshold
        peaks, _ = signal.find_peaks(avg_spectrum, height=0.1, distance=int(0.5 / (unique_freqs[1] - unique_freqs[0])))
    
    suggested_ranges = []
    if len(peaks) > 0:
        peak_freqs = unique_freqs[peaks]
        peak_amplitudes = avg_spectrum[peaks]
        
        # Sort peaks by amplitude
        sorted_indices = np.argsort(-peak_amplitudes)
        peak_freqs = peak_freqs[sorted_indices]
        peak_amplitudes = peak_amplitudes[sorted_indices]
        
        # Generate frequency ranges around the top peaks
        for i, (freq, amp) in enumerate(zip(peak_freqs, peak_amplitudes)):
            if i >= 3:  # Limit to top 3 suggestions
                break
            
            # Create a range around the peak
            lower = max(0.1, freq - 0.2)
            upper = freq + 0.2
            
            suggested_ranges.append({
                'freq_min': float(lower),
                'freq_max': float(upper),
                'peak_freq': float(freq),
                'amplitude': float(amp),
                'description': f"Peak at {freq:.2f} Hz"
            })
    
    # If no clear peaks, suggest ranges based on frequency energy distribution
    if len(suggested_ranges) == 0:
        # Low frequency range (0.1-0.5 Hz) - often for breathing, slow movements
        suggested_ranges.append({
            'freq_min': 0.1,
            'freq_max': 0.5,
            'description': "Low frequency range (breathing, slow movements)"
        })
        
        # Medium frequency range (0.5-2.0 Hz) - often for heartbeats, moderate movements
        suggested_ranges.append({
            'freq_min': 0.5,
            'freq_max': 2.0,
            'description': "Medium frequency range (heartbeats, moderate movements)"
        })
    
    # Save analysis results and metadata to JSON
    analysis_data = {
        'video': {
            'path': video_path,
            'fps': float(fps),
            'dimensions': [width, height],
            'frame_count': n_frames
        },
        'analysis': {
            'sampling_rate': sampling_rate,
            'analysis_fps': float(analysis_fps),
            'analyzed_frames': len(sampled_frames),
            'tracked_points': len(valid_trajectories)
        },
        'suggested_ranges': suggested_ranges,
        'visualization_paths': visualization_paths,
        'timestamp': import_time_module().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Calculate a hash of the input video for cache validation
    video_hash = get_file_hash(video_path)
    analysis_data['video']['hash'] = video_hash
    
    # Save analysis data
    metadata_path = os.path.join(output_dir, f"{base_name}_frequency_analysis.json")
    with open(metadata_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    print(f"Suggested frequency ranges:")
    for i, range_info in enumerate(suggested_ranges):
        desc = range_info.get('description', '')
        if 'peak_freq' in range_info:
            print(f"  {i+1}. {range_info['freq_min']:.2f}-{range_info['freq_max']:.2f} Hz (peak at {range_info['peak_freq']:.2f} Hz) - {desc}")
        else:
            print(f"  {i+1}. {range_info['freq_min']:.2f}-{range_info['freq_max']:.2f} Hz - {desc}")
    
    return {
        'metadata_path': metadata_path,
        'suggested_ranges': suggested_ranges,
        'visualization_paths': visualization_paths
    }

def load_analysis_results(metadata_path):
    """
    Load previously generated frequency analysis results.
    
    Parameters:
    -----------
    metadata_path : str
        Path to the analysis metadata JSON file
    
    Returns:
    --------
    dict
        Dictionary containing the analysis results
    """
    try:
        with open(metadata_path, 'r') as f:
            analysis_data = json.load(f)
        
        # Verify that all visualization files exist
        for viz_type, viz_path in analysis_data['visualization_paths'].items():
            if isinstance(viz_path, dict):
                for freq, path in viz_path.items():
                    if not os.path.exists(path):
                        print(f"Warning: Visualization file not found: {path}")
            elif not os.path.exists(viz_path):
                print(f"Warning: Visualization file not found: {viz_path}")
        
        return analysis_data
    except Exception as e:
        print(f"Error loading analysis results: {e}")
        return None

def validate_analysis_cache(metadata_path, video_path):
    """
    Check if cached analysis results are valid for the given video.
    
    Parameters:
    -----------
    metadata_path : str
        Path to the analysis metadata JSON file
    video_path : str
        Path to the video file to validate against
    
    Returns:
    --------
    bool
        True if the cache is valid, False otherwise
    """
    try:
        # Check if metadata file exists
        if not os.path.exists(metadata_path):
            return False
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            analysis_data = json.load(f)
        
        # Check if the video path matches
        if analysis_data['video']['path'] != video_path:
            # If paths don't match, check the hash
            video_hash = get_file_hash(video_path)
            if analysis_data['video'].get('hash') != video_hash:
                return False
        
        # Check if all visualization files exist
        for viz_type, viz_path in analysis_data['visualization_paths'].items():
            if isinstance(viz_path, dict):
                for freq, path in viz_path.items():
                    if not os.path.exists(path):
                        return False
            elif not os.path.exists(viz_path):
                return False
        
        return True
    except Exception as e:
        print(f"Error validating analysis cache: {e}")
        return False

def get_file_hash(file_path, block_size=65536):
    """Calculate a hash for a file to use for cache validation."""
    if not os.path.exists(file_path):
        return None
    
    hash_obj = hashlib.md5()
    with open(file_path, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            hash_obj.update(block)
    return hash_obj.hexdigest()

def import_time_module():
    """Import time module (used to avoid global imports)."""
    import time
    return time
