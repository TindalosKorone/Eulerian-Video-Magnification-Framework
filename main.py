import argparse
import torch
import cv2
import video_handler as vh
import motion_amplifier as ma
import phase_amplifier as pa
from tqdm import tqdm
import gc
import os
import tempfile
import stabilizer

def main():
    parser = argparse.ArgumentParser(description='Magnify motion in a video.')
    parser.add_argument('video_path', type=str, help='Path to the input video file.')
    parser.add_argument('output_path', type=str, help='Path to the output video file.')
    parser.add_argument('--alpha', type=float, default=10.0, help='Motion amplification factor.')
    parser.add_argument('--low_freq', type=float, default=0.4, help='Low frequency cutoff for the bandpass filter.')
    parser.add_argument('--high_freq', type=float, default=3.0, help='High frequency cutoff for the bandpass filter.')
    parser.add_argument('--levels', type=int, default=3, help='Number of levels in the Laplacian pyramid.')
    parser.add_argument('--chunk_size', type=int, default=20, help='Number of frames to process at a time.')
    parser.add_argument('--overlap', type=int, default=8, help='Number of overlapping frames between chunks.')
    parser.add_argument('--spatial_blur_sigma', type=float, default=0, help='Standard deviation for Gaussian blur of the motion signal (linear method only). 0 to disable.')
    parser.add_argument('--stabilize', action='store_true', help='Perform video stabilization before motion magnification.')
    parser.add_argument('--method', type=str, default='linear', choices=['linear', 'phase'], help='Motion magnification method to use.')
    
    args = parser.parse_args()

    if args.overlap >= args.chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_video_path = args.video_path
    temp_file_handle, temp_stabilized_path = None, None

    if args.stabilize:
        print("Stabilization enabled. This may take a while...")
        temp_file_handle, temp_stabilized_path = tempfile.mkstemp(suffix=".mp4")
        os.close(temp_file_handle) # Close the handle so the stabilizer can write to the file
        stabilizer.stabilize_video(args.video_path, temp_stabilized_path)
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
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))

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
                
                if args.method == 'linear':
                    result_tensor = ma.magnify_motion(tensor, fps, args.low_freq, args.high_freq, args.alpha, args.levels, args.spatial_blur_sigma)
                else: # phase
                    result_tensor = pa.magnify_phase(tensor, fps, args.low_freq, args.high_freq, args.alpha)

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
            if args.method == 'linear':
                result_tensor = ma.magnify_motion(tensor, fps, args.low_freq, args.high_freq, args.alpha, args.levels, args.spatial_blur_sigma)
            else: # phase
                result_tensor = pa.magnify_phase(tensor, fps, args.low_freq, args.high_freq, args.alpha)
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

    print(f"Video written to {args.output_path}")

if __name__ == '__main__':
    main()
