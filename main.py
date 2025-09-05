import argparse
import torch
import cv2
import video_handler as vh
import motion_amplifier as ma
from tqdm import tqdm
import gc
import os
import tempfile
import stabilizer
import frequency_analyzer
import cache_manager
import sys

# Preset configurations
PRESETS = {
    'subtle': {
        'amplify': 5.0,
        'freq_range': (0.5, 2.0),
        'pyramid_levels': 3,
        'blur': 0
    },
    'medium': {
        'amplify': 10.0,
        'freq_range': (0.4, 3.0),
        'pyramid_levels': 3,
        'blur': 0
    },
    'extreme': {
        'amplify': 20.0,
        'freq_range': (0.2, 4.0),
        'pyramid_levels': 4,
        'blur': 1.0
    },
    'pulse': {  # For heartbeats, subtle movements
        'amplify': 15.0,
        'freq_range': (0.8, 2.0),  # Typical pulse range
        'pyramid_levels': 3,
        'blur': 0.5
    },
    'breathing': {  # For respiratory movements
        'amplify': 10.0,
        'freq_range': (0.1, 0.5),  # Typical breathing range
        'pyramid_levels': 3,
        'blur': 0.8
    }
}

def main():
    # Create the main parser
    parser = argparse.ArgumentParser(
        description='Motion Magnification Framework',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('input', type=str, 
                        help='Path to the input video file')
    parser.add_argument('output', type=str, nargs='?', default=None,
                        help='Path to the output video file (optional for analysis-only mode)')
    
    
    # Operation mode options
    mode_group = parser.add_argument_group('Operation Modes')
    mode_group.add_argument('--analyze-only', action='store_true',
                          help='Only perform frequency analysis without magnification')
    mode_group.add_argument('--use-analysis', type=str, metavar='PATH',
                          help='Use existing frequency analysis results from the specified JSON file')
    mode_group.add_argument('--no-cache', action='store_true',
                          help='Disable caching (by default, caching is enabled)')
    mode_group.add_argument('--suggest-params', action='store_true',
                          help='Suggest parameters based on video analysis and exit')
    
    # Preset configuration
    parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                        help='Use a predefined parameter preset for common scenarios')
    
    # Create parameter groups for better organization
    basic_group = parser.add_argument_group('Basic Parameters')
    basic_group.add_argument('--amplify', type=float, default=10.0,
                          help='Motion amplification factor')
    
    filter_group = parser.add_argument_group('Filtering Parameters')
    filter_group.add_argument('--freq-min', type=float, default=0.4,
                           help='Minimum frequency to amplify (Hz)')
    filter_group.add_argument('--freq-max', type=float, default=3.0,
                           help='Maximum frequency to amplify (Hz)')
    filter_group.add_argument('--blur', type=float, default=0,
                           help='Spatial blur strength (0 to disable)')
    filter_group.add_argument('--motion-threshold', type=float, default=0,
                           help='Minimum motion magnitude to amplify (0 to disable)')
    
    pyramid_group = parser.add_argument_group('Pyramid Parameters')
    pyramid_group.add_argument('--levels', type=int, default=3,
                            help='Number of pyramid levels')
    
    processing_group = parser.add_argument_group('Processing Parameters')
    processing_group.add_argument('--chunk-size', type=int, default=20,
                               help='Number of frames to process at once')
    processing_group.add_argument('--overlap', type=int, default=8,
                               help='Number of overlapping frames between chunks')
    
    enhancement_group = parser.add_argument_group('Enhancements')
    enhancement_group.add_argument('--stabilize', action='store_true',
                                help='Stabilize video before magnification')
    enhancement_group.add_argument('--stabilize-radius', type=int, default=None,
                                help='Stabilization smoothing radius in frames (default: fps)')
    enhancement_group.add_argument('--stabilize-strength', type=float, default=0.95,
                                help='Stabilization strength (0.0-1.0)')
    enhancement_group.add_argument('--adaptive', action='store_true',
                                help='Use adaptive amplification (stronger for larger structures)')
    enhancement_group.add_argument('--bilateral', action='store_true',
                                help='Use bilateral filtering (preserves edges better)')
    enhancement_group.add_argument('--color-stabilize', action='store_true',
                                help='Stabilize colors to reduce flickering')
    enhancement_group.add_argument('--multiband', action='store_true',
                                help='Process frequency range in multiple bands')
    enhancement_group.add_argument('--keep-temp', action='store_true',
                                help='Keep temporary files (like stabilized video)')
    
    # Analysis parameters
    analysis_group = parser.add_argument_group('Analysis Parameters')
    analysis_group.add_argument('--analysis-dir', type=str, default=None,
                              help='Directory to save analysis results (default: same as output)')
    analysis_group.add_argument('--sampling-rate', type=float, default=0.5,
                              help='Fraction of frames to analyze (0.0-1.0)')
    analysis_group.add_argument('--max-points', type=int, default=200,
                              help='Maximum number of feature points to track per frame')
    analysis_group.add_argument('--downsample-factor', type=int, default=1,
                              help='Spatial downsample factor for frequency maps (higher values use less memory)')
    analysis_group.add_argument('--max-freq-bands', type=int, default=20,
                              help='Maximum number of frequency bands to analyze (higher values use more memory)')
    analysis_group.add_argument('--skip-visualizations', action='store_true',
                              help='Skip generating visualization images to speed up analysis')
    
    args = parser.parse_args()

    # Check for valid argument combinations
    if args.analyze_only and args.output is not None:
        print("Warning: Output path is ignored in analysis-only mode")
    
    if not args.analyze_only and args.output is None:
        parser.error("Output path is required unless --analyze-only is specified")
    
    # Initialize cache manager
    cm = cache_manager.CacheManager()
    
    # Apply preset if specified
    if args.preset:
        preset = PRESETS[args.preset]
        args.amplify = preset['amplify']
        args.freq_min = preset['freq_range'][0]
        args.freq_max = preset['freq_range'][1]
        args.levels = preset['pyramid_levels']
        args.blur = preset['blur']
        print(f"Using '{args.preset}' preset with: amplify={args.amplify}, "
              f"freq_range=({args.freq_min}-{args.freq_max}Hz), "
              f"levels={args.levels}, blur={args.blur}")

    if args.overlap >= args.chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_video_path = args.input
    temp_stabilized_path = None
    analysis_results = None
    suggested_ranges = []

    # Determine the output directory for saving analysis results
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
    else:
        output_dir = os.path.dirname(os.path.abspath(args.input))
    
    if args.analysis_dir:
        analysis_dir = args.analysis_dir
    else:
        analysis_dir = output_dir
        
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    
    # Check if we should use an existing analysis
    if args.use_analysis:
        if os.path.exists(args.use_analysis):
            analysis_results = frequency_analyzer.load_analysis_results(args.use_analysis)
            if analysis_results:
                print(f"Loaded analysis results from: {args.use_analysis}")
                suggested_ranges = analysis_results.get('suggested_ranges', [])
                
                # Display the suggested frequency ranges
                print("Suggested frequency ranges based on analysis:")
                for i, range_info in enumerate(suggested_ranges):
                    desc = range_info.get('description', '')
                    if 'peak_freq' in range_info:
                        print(f"  {i+1}. {range_info['freq_min']:.2f}-{range_info['freq_max']:.2f} Hz (peak at {range_info['peak_freq']:.2f} Hz) - {desc}")
                    else:
                        print(f"  {i+1}. {range_info['freq_min']:.2f}-{range_info['freq_max']:.2f} Hz - {desc}")
            else:
                print(f"Error: Could not load analysis results from {args.use_analysis}")
                sys.exit(1)
        else:
            print(f"Error: Analysis file {args.use_analysis} does not exist")
            sys.exit(1)
    
    # If no existing analysis and suggest-params or analyze-only mode, perform analysis
    if not analysis_results and (args.analyze_only or args.suggest_params):
        # Check if we have cached analysis results
        cached_analysis_path = None
        if not args.no_cache:
            analysis_params = {
                'sampling_rate': args.sampling_rate,
                'max_points': args.max_points,
                'downsample_factor': args.downsample_factor,
                'max_freq_bands': args.max_freq_bands,
                'skip_visualizations': args.skip_visualizations
            }
            cached_analysis_path = cm.get_analysis_results_path(args.input, analysis_params)
        
        if cached_analysis_path:
            # Use cached analysis
            analysis_results = frequency_analyzer.load_analysis_results(cached_analysis_path)
            print(f"Using cached frequency analysis from: {cached_analysis_path}")
            suggested_ranges = analysis_results.get('suggested_ranges', [])
        else:
            # Perform new analysis
            print("Performing frequency analysis...")
            analysis_output = frequency_analyzer.analyze_video_frequencies(
                args.input, 
                output_dir=analysis_dir,
                sampling_rate=args.sampling_rate,
                max_points=args.max_points,
                downsample_factor=args.downsample_factor,
                max_freq_bands=args.max_freq_bands,
                generate_visualizations=not args.skip_visualizations
            )
            
            # Cache the analysis results
            if 'metadata_path' in analysis_output:
                analysis_results = frequency_analyzer.load_analysis_results(analysis_output['metadata_path'])
                analysis_params = {
                    'sampling_rate': args.sampling_rate,
                    'max_points': args.max_points,
                    'downsample_factor': args.downsample_factor,
                    'max_freq_bands': args.max_freq_bands,
                    'skip_visualizations': args.skip_visualizations
                }
                cm.add_analysis_results(
                    args.input, 
                    analysis_output['metadata_path'],
                    analysis_output['visualization_paths'],
                    analysis_params
                )
                suggested_ranges = analysis_output.get('suggested_ranges', [])
        
        # If only suggesting parameters or analyze-only mode, exit here
        if args.suggest_params:
            if suggested_ranges:
                # Use the first suggested range as the recommended parameters
                best_range = suggested_ranges[0]
                print("\nRecommended parameters:")
                print(f"  --freq-min {best_range['freq_min']:.2f} --freq-max {best_range['freq_max']:.2f}")
                if 'peak_freq' in best_range:
                    print(f"  Motion peak detected at {best_range['peak_freq']:.2f} Hz")
            else:
                print("\nNo clear frequency peaks detected. Using default parameters is recommended.")
            sys.exit(0)
            
        if args.analyze_only:
            print("\nFrequency analysis complete. Run again with normal mode to magnify motion.")
            sys.exit(0)
            
    # Stabilization (with cache support)
    if args.stabilize:
        # Check if we have a cached stabilized version
        cached_stabilized_path = None
        if not args.no_cache:
            stabilize_params = {
                'radius': args.stabilize_radius,
                'strength': args.stabilize_strength
            }
            cached_stabilized_path = cm.get_stabilized_video_path(args.input, stabilize_params)
        
        if cached_stabilized_path:
            # Use cached stabilized video
            input_video_path = cached_stabilized_path
            temp_stabilized_path = cached_stabilized_path
            print(f"Using cached stabilized video: {cached_stabilized_path}")
        else:
            print("Stabilization enabled. This may take a while...")
            # Get output directory
            if not output_dir:  # If output is in current directory
                output_dir = os.getcwd()
                
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Generate stabilized video filename in the same directory as output
            input_filename = os.path.basename(args.input)
            input_name, input_ext = os.path.splitext(input_filename)
            temp_stabilized_path = os.path.join(output_dir, f"{input_name}_stabilized{input_ext}")
            
            # Perform stabilization
            stabilizer.stabilize_video(args.input, temp_stabilized_path, 
                                     smoothing_radius=args.stabilize_radius,
                                     smoothing_strength=args.stabilize_strength)
            input_video_path = temp_stabilized_path
            print(f"Stabilized video saved to: {temp_stabilized_path}")
            
            # Cache the stabilized video
            if not args.no_cache:
                stabilize_params = {
                    'radius': args.stabilize_radius,
                    'strength': args.stabilize_strength
                }
                cm.add_stabilized_video(args.input, temp_stabilized_path, stabilize_params)

    vidcap = cv2.VideoCapture(input_video_path)
    if not vidcap.isOpened():
        raise IOError(f"Video at {input_video_path} could not be opened.")
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frames_buffer = []
    processed_frames_count = 0
    write_buffer = []

    with tqdm(total=frame_count, desc="Processing video") as pbar:
        while True:
            success, frame = vidcap.read()
            if not success:
                break
            frames_buffer.append(frame)

            if len(frames_buffer) == args.chunk_size:
                tensor = vh.frames_to_tensor(frames_buffer, device)
                
                result_tensor = ma.magnify_motion(
                    tensor, fps, args.freq_min, args.freq_max, 
                    args.amplify, args.levels, args.blur,
                    args.motion_threshold, args.adaptive,
                    args.bilateral, args.color_stabilize, args.multiband
                )

                result_frames = vh.tensor_to_frames(result_tensor)

                # Determine the frames to write, considering overlap
                start_index = args.overlap // 2 if processed_frames_count > 0 else 0
                end_index = args.chunk_size - (args.overlap // 2)
                
                for i in range(start_index, end_index):
                    out.write(result_frames[i])
                
                pbar.update(end_index - start_index)
                processed_frames_count += (end_index - start_index)

                # Slide the buffer
                frames_buffer = frames_buffer[args.chunk_size - args.overlap:]

                # Clean up memory
                del tensor, result_tensor, result_frames
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                else:
                    gc.collect()

        # Process the remaining frames in the buffer
        if len(frames_buffer) > 0:
            # Pad the buffer if it's smaller than the chunk size
            while len(frames_buffer) < args.chunk_size and len(frames_buffer) > 0:
                 frames_buffer.append(frames_buffer[-1]) # Simple padding by repeating the last frame

            tensor = vh.frames_to_tensor(frames_buffer, device)
            result_tensor = ma.magnify_motion(
                tensor, fps, args.freq_min, args.freq_max, 
                args.amplify, args.levels, args.blur,
                args.motion_threshold, args.adaptive,
                args.bilateral, args.color_stabilize, args.multiband
            )
                    
            result_frames = vh.tensor_to_frames(result_tensor)
            
            remaining_to_write = frame_count - processed_frames_count
            start_index = args.overlap // 2
            
            for i in range(start_index, min(start_index + remaining_to_write, len(result_frames))):
                out.write(result_frames[i])
            
            pbar.update(min(start_index + remaining_to_write, len(result_frames)) - start_index)
            
            # Clean up memory
            del tensor, result_tensor, result_frames
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            else:
                gc.collect()

    vidcap.release()
    out.release()

    # Handle temporary files
    if temp_stabilized_path and not args.keep_temp and not cached_stabilized_path:
        print(f"Cleaning up temporary file: {temp_stabilized_path}")
        os.remove(temp_stabilized_path)
    elif temp_stabilized_path and args.keep_temp:
        print(f"Temporary stabilized video kept at: {temp_stabilized_path}")
        
    # Clean up old cache files periodically
    cm.clean_cache()

    print(f"Video processing complete! Output saved to: {args.output}")
    print("\nTip: If you want to try different settings, try using presets like:")
    print(f"  python main.py {args.input} {args.output} --preset pulse")
    print("Or customize parameters directly:")
    print(f"  python main.py {args.input} {args.output} --amplify 15 --freq-min 0.8 --freq-max 2.0")
    print("\nAdvanced options for noise reduction:")
    print(f"  python main.py {args.input} {args.output} --blur 0.5 --motion-threshold 0.02 --adaptive")
    print("\nAdvanced enhancement options:")
    print(f"  python main.py {args.input} {args.output} --bilateral --color-stabilize --multiband")
    print("\nVideo stabilization options:")
    print(f"  python main.py {args.input} {args.output} --stabilize --stabilize-radius 15 --stabilize-strength 0.8 --keep-temp")
    print("\nFrequency analysis options:")
    print(f"  python main.py {args.input} --analyze-only")
    print(f"  python main.py {args.input} --suggest-params")
    print(f"  python main.py {args.input} {args.output} --use-analysis path/to/analysis.json")
    print("\nCache management options:")
    print(f"  python main.py {args.input} {args.output} --no-cache  # 禁用缓存（默认启用）")
    print("\nPerformance optimization options:")
    print(f"  python main.py {args.input} --analyze-only --sampling-rate 0.2 --max-points 100")
    print(f"  python main.py {args.input} --analyze-only --downsample-factor 2 --max-freq-bands 10 --skip-visualizations")

if __name__ == '__main__':
    main()
