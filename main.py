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
    parser.add_argument('output', type=str, 
                        help='Path to the output video file')
    
    
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
    enhancement_group.add_argument('--adaptive', action='store_true',
                                help='Use adaptive amplification (stronger for larger structures)')
    enhancement_group.add_argument('--bilateral', action='store_true',
                                help='Use bilateral filtering (preserves edges better)')
    enhancement_group.add_argument('--color-stabilize', action='store_true',
                                help='Stabilize colors to reduce flickering')
    enhancement_group.add_argument('--multiband', action='store_true',
                                help='Process frequency range in multiple bands')
    
    args = parser.parse_args()

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
    temp_file_handle, temp_stabilized_path = None, None

    if args.stabilize:
        print("Stabilization enabled. This may take a while...")
        temp_file_handle, temp_stabilized_path = tempfile.mkstemp(suffix=".mp4")
        os.close(temp_file_handle) # Close the handle so the stabilizer can write to the file
        stabilizer.stabilize_video(args.input, temp_stabilized_path)
        input_video_path = temp_stabilized_path
        print(f"Stabilized video saved temporarily to {temp_stabilized_path}")

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

    if temp_stabilized_path:
        print(f"Cleaning up temporary file: {temp_stabilized_path}")
        os.remove(temp_stabilized_path)

    print(f"Video processing complete! Output saved to: {args.output}")
    print("\nTip: If you want to try different settings, try using presets like:")
    print(f"  python main.py {args.input} {args.output} --preset pulse")
    print("Or customize parameters directly:")
    print(f"  python main.py {args.input} {args.output} --amplify 15 --freq-min 0.8 --freq-max 2.0")
    print("\nAdvanced options for noise reduction:")
    print(f"  python main.py {args.input} {args.output} --blur 0.5 --motion-threshold 0.02 --adaptive")
    print("\nAdvanced enhancement options:")
    print(f"  python main.py {args.input} {args.output} --bilateral --color-stabilize --multiband")

if __name__ == '__main__':
    main()
