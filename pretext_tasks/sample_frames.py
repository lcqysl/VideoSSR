import cv2
import os
import json
import concurrent.futures
from decord import VideoReader, cpu
from tqdm import tqdm


NUM_FRAMES_TO_SAMPLE = 48
NUM_WORKER = 96
VIDEO_DIRS = [
    "path1/to/videos",
    "path2/to/videos"
]
OUTPUT_BASE_DIR = f"path/to/{NUM_FRAMES_TO_SAMPLE}frames"
JSON_OUTPUT_PATH = f"path/to/{NUM_FRAMES_TO_SAMPLE}frames_fps.json"

MAX_VIDEOS_TO_PROCESS = None 

def sample_frames_with_decord(video_path, output_dir, num_frames=NUM_FRAMES_TO_SAMPLE):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        start_frame = int(total_frames * 0.05)
        end_frame = int(total_frames * 0.95)
        effective_total_frames = end_frame - start_frame

        if effective_total_frames < num_frames:
            return None, None

        os.makedirs(output_dir, exist_ok=True)
        
        interval = effective_total_frames // num_frames
        frame_indices = [start_frame + i * interval for i in range(num_frames)]
        
        frames = vr.get_batch(frame_indices).asnumpy()

        for i, frame in enumerate(frames):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            output_path = os.path.join(output_dir, f"{i}.jpg")
            cv2.imwrite(output_path, frame_bgr)

        fps = vr.get_avg_fps()
        original_duration = total_frames / fps
        sampling_fps = num_frames / original_duration if original_duration > 0 else 0
        
        return sampling_fps
    except Exception as e:
        return None

def process_video_wrapper(video_info):
    video_key, video_path, output_dir_for_video = video_info
    sampling_fps = sample_frames_with_decord(video_path, output_dir_for_video)
    return video_key, sampling_fps

def find_all_videos(root_dirs):
    video_paths = []
    for root_dir in root_dirs:
        if not os.path.isdir(root_dir):
            continue
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith('.mp4'):
                    full_path = os.path.join(dirpath, filename)
                    video_paths.append((root_dir, full_path))
    return video_paths

def main():
    video_files_to_process = find_all_videos(VIDEO_DIRS)
    
    if not video_files_to_process:
        return
        
    if MAX_VIDEOS_TO_PROCESS is not None and MAX_VIDEOS_TO_PROCESS > 0:
        video_files_to_process = video_files_to_process[:MAX_VIDEOS_TO_PROCESS]

    tasks = []
    for root_dir, video_path in video_files_to_process:
        relative_path = os.path.relpath(video_path, root_dir)
        video_key = os.path.splitext(relative_path)[0]
        output_dir_for_video = os.path.join(OUTPUT_BASE_DIR, video_key)
        
        tasks.append((video_key, video_path, output_dir_for_video))

    fps_data = {}
    
    max_workers = NUM_WORKER

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_video_wrapper, tasks), total=len(tasks), desc="Sampling Videos"))

    successful_count = 0
    for video_key, sampling_fps in results:
        if sampling_fps is not None:
            fps_data[video_key] = sampling_fps
            successful_count += 1

    os.makedirs(os.path.dirname(JSON_OUTPUT_PATH), exist_ok=True)
    with open(JSON_OUTPUT_PATH, 'w') as f:
        json.dump(fps_data, f, indent=4)
        

if __name__ == "__main__":
    main()