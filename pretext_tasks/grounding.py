import os
import datasets
import random
import json
from PIL import Image, ImageOps, ImageDraw, ImageFilter,ImageEnhance
import numpy as np


PERTURBATION_TYPE = "ChannelSwap"
START_INDEX = 0
END_INDEX = 3000 

SYSTEM_PROMPT = """Based on this video, provide the answer directly."""

PERTURBATION_DETAILS = {
    "Grayscale": {"description": "the video becomes black and white", "suffix": "grayscale"},
    "Invert": {"description": "the colors in the video are inverted", "suffix": "invert"},
    "ChannelSwap": {"description": "the red and blue color channels in the video are swapped", "suffix": "channelswap"},
    "Rotate90": {"description": "the video is rotated 90 degrees clockwise", "suffix": "rotate90"},
    "Rotate180": {"description": "the video is rotated 180 degrees", "suffix": "rotate180"},
    "Rotate270": {"description": "the video is rotated 270 degrees clockwise", "suffix": "rotate270"},
    "Noise": {"description": "Gaussian noise is added to the video", "suffix": "noise"},
    "Shuffle": {"description": "the frames are shuffled, this means the order of events is scrambled, making the action appear illogical and chaotic", "suffix": "shuffle"},
    "Swap": {"description": "the clip is replaced by a clip from another video", "suffix": "swap"},
    "ZoomIn": {"description": "the video is zoomed in", "suffix": "zoomin"},
    "ZoomOut": {"description": "the video is zoomed out", "suffix": "zoomout"},
    "Mirror": {"description": "The video is mirrored horizontally", "suffix": "mirror"},
    "Blur": {"description": "the video becomes blurry or out of focus", "suffix": "blur"},
    "Saturation": {"description": "the colors in the video become oversaturated and unnaturally vibrant", "suffix": "saturation"},
    "StutterHold": {"description": "the video appears to freeze and stutter on a few frames, this means instead of playing smoothly, the video repeatedly freezes on a single frame before jumping to the next.", "suffix": "stutterhold"},
}

if PERTURBATION_TYPE not in PERTURBATION_DETAILS:
    raise ValueError(f"Invalid PERTURBATION_TYPE. Choose from {list(PERTURBATION_DETAILS.keys())}")

QUESTION_TEXT_TEMPLATE = "In a segment of this video, {description}. Your task is to identify the precise time interval of this change. Please only provide the start and end times in seconds, formatted as <start_time>-<end_time> (e.g., '14.5-26.2')."
PERTURB_INFO = PERTURBATION_DETAILS[PERTURBATION_TYPE]
QUESTION_TEXT = QUESTION_TEXT_TEMPLATE.format(description=PERTURB_INFO["description"])
FILENAME_SUFFIX = PERTURB_INFO["suffix"]

MIN_FRAMES_NEEDED = 48

SPLIT_NAME = "train"
VIDEO_BASE_DIR = f"path/to/{MIN_FRAMES_NEEDED}frames"
FPS_JSON_PATH = f"path/to/{MIN_FRAMES_NEEDED}frames_fps.json"
LOCAL_SAVE_DIR = "path/to/parquet"
NAME = f"{FILENAME_SUFFIX}"
MAX_PIXELS = 256

def apply_grayscale(image): return ImageOps.grayscale(image).convert('RGB')
def apply_invert(image): return ImageOps.invert(image.convert('RGB'))
def apply_channel_swap(image): r, g, b = image.convert('RGB').split(); return Image.merge('RGB', (b, g, r))
def apply_rotate_90(image): return image.rotate(-90, expand=True)
def apply_rotate_180(image): return image.rotate(180)
def apply_rotate_270(image): return image.rotate(90, expand=True)
def apply_noise(image):
    image_np = np.array(image.convert('RGB'))
    intensity = random.uniform(35, 70)
    noise = np.random.normal(0, intensity, image_np.shape)
    noisy_image_np = np.clip(image_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image_np)

def apply_zoom_in(image):
    scale = random.uniform(1.3, 1.5)
    width, height = image.size
    new_width, new_height = width / scale, height / scale
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image.resize(image.size, Image.LANCZOS)

def apply_zoom_out(image):
    scale = random.uniform(1.2, 1.4)
    width, height = image.size
    new_width, new_height = int(width / scale), int(height / scale)
    shrunk_image = image.resize((new_width, new_height), Image.LANCZOS)
    background = Image.new('RGB', image.size, (0, 0, 0))
    paste_x = (width - new_width) // 2
    paste_y = (height - new_height) // 2
    background.paste(shrunk_image, (paste_x, paste_y))
    return background

def apply_mirror(image):
    return ImageOps.mirror(image)

def apply_blur(image):
    return image.filter(ImageFilter.GaussianBlur(radius=2))

def apply_saturation(image):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(2.0)

perturb_functions = {
    "Grayscale": apply_grayscale, "Invert": apply_invert, "ChannelSwap": apply_channel_swap,
    "Rotate90": apply_rotate_90, "Rotate180": apply_rotate_180, "Rotate270": apply_rotate_270,
    "Noise": apply_noise,
    "ZoomIn": apply_zoom_in, "ZoomOut": apply_zoom_out,
    "Mirror": apply_mirror,     
    "Blur": apply_blur,
    "Saturation": apply_saturation,
}


def make_map_fn(split, video_base_dir, fps_data, perturbation_type, all_video_ids=None):
    
    process_perturbation = perturb_functions.get(perturbation_type)
    
    def process_fn(batch):
        results = []
        for idx, video_id in enumerate(batch["video"]):
            video_path = os.path.join(video_base_dir, video_id)
            sampling_fps = fps_data.get(video_id)

            if sampling_fps is None or not os.path.isdir(video_path): continue

            try:
                original_frames = [f for f in os.listdir(video_path) if f.endswith(".jpg") and "_" not in f]
                frames = sorted(original_frames, key=lambda x: int(x.split(".")[0]))
            except ValueError: continue
            
            t = random.randint(8, 16)
            
            total_frames_to_read = MIN_FRAMES_NEEDED

            if len(frames) < total_frames_to_read: continue
            
            source_frames = frames[:total_frames_to_read]
            
            max_start_pos = len(source_frames) - t
            if max_start_pos < 0: continue
            start_frame_index = random.randint(0, max_start_pos)
            end_frame_index = start_frame_index + t - 1
            
            modified_frame_paths = []
            final_video_paths = []
            final_segment_len = t
            
            
            if perturbation_type == "Shuffle":
                frames_to_shuffle = source_frames[start_frame_index : end_frame_index + 1]
                segment_paths = [os.path.join(video_path, f) for f in frames_to_shuffle]
                random.shuffle(segment_paths)
                modified_frame_paths = segment_paths

            elif perturbation_type == "Swap":
                possible_swap_ids = [vid for vid in all_video_ids if vid != video_id]
                swap_video_id = random.choice(possible_swap_ids)
                swap_video_path = os.path.join(video_base_dir, swap_video_id)
                
                if not os.path.isdir(swap_video_path): continue

                frames_to_replace_names = source_frames[start_frame_index : end_frame_index + 1]
                swap_paths = [os.path.join(swap_video_path, f) for f in frames_to_replace_names]
                
                if not all(os.path.exists(p) for p in swap_paths): continue
                
                modified_frame_paths = swap_paths
            
            elif perturbation_type == "StutterHold":
                segment_to_modify = source_frames[start_frame_index : end_frame_index + 1]
                segment_paths = [os.path.join(video_path, f) for f in segment_to_modify]
                n_hold_frames = random.choice([2, 3, 4, 5])
                
                indices = np.linspace(0, len(segment_paths) - 1, n_hold_frames, dtype=int)
                selected_frame_paths = [segment_paths[i] for i in indices]
                
                base_reps = t // n_hold_frames
                remainder = t % n_hold_frames
                
                stuttered_paths = []
                for i, frame_path in enumerate(selected_frame_paths):
                    reps = base_reps + 1 if i < remainder else base_reps
                    stuttered_paths.extend([frame_path] * reps)
                
                modified_frame_paths = stuttered_paths

            else:
                if process_perturbation is None: continue
                
                frames_to_modify_names = source_frames[start_frame_index : start_frame_index + t]
                for frame_filename in frames_to_modify_names:
                    original_path = os.path.join(video_path, frame_filename)
                    base_name, ext = os.path.splitext(frame_filename)
                    new_filename = f"{base_name}_{FILENAME_SUFFIX}{ext}"
                    new_path = os.path.join(video_path, new_filename)
                    
                    if not os.path.exists(new_path):
                        try:
                            with Image.open(original_path) as img:
                                modified_img = process_perturbation(img)
                                modified_img.save(new_path)
                        except Exception as e:
                            print(f"Warning: Failed to process {original_path}. Error: {e}")
                            modified_frame_paths = []
                            break
                    modified_frame_paths.append(new_path)
                
            if len(modified_frame_paths) == t:
                prefix_paths = [os.path.join(video_path, f) for f in source_frames[:start_frame_index]]
                suffix_paths = [os.path.join(video_path, f) for f in source_frames[start_frame_index + t:]]
                final_video_paths = prefix_paths + modified_frame_paths + suffix_paths
            else:
                continue

            if not final_video_paths: continue

            end_frame_index = start_frame_index + final_segment_len
            start_time = start_frame_index / sampling_fps
            end_time = end_frame_index / sampling_fps
            ground_truth_answer = f"{start_time:.1f}-{end_time:.1f}"

            results.append({
                "data_source": "grounding",
                "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"<video>\n{QUESTION_TEXT}"}],
                "videos": [{"video": final_video_paths, "sample_fps": sampling_fps, "max_pixels": MAX_PIXELS * MAX_PIXELS, "min_pixels": 128 * 128}],
                "ability": "video_understanding",
                "reward_model": {"style": "rule", "ground_truth": ground_truth_answer},
                "extra_info": { "split": split, "index": idx, "answer": ground_truth_answer, "question": QUESTION_TEXT, "video_id": video_id,
                                "sampling_fps": sampling_fps, "perturbation_type": perturbation_type, "perturbed_frames_count": final_segment_len, "start_frame_index": start_frame_index},
            })

        if not results: return {}
        return {k: [item[k] for item in results] for k in results[0].keys()}

    return process_fn


if __name__ == "__main__":
    print(f"Loading FPS data from: {FPS_JSON_PATH}")
    try:
        with open(FPS_JSON_PATH, "r") as f:
            fps_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading FPS file: {e}")
        exit()

    video_ids = list(fps_data.keys())

    if not video_ids:
        print("Error: No video entries found.")
        exit()

    if START_INDEX >= len(video_ids) or END_INDEX > len(video_ids) or START_INDEX >= END_INDEX:
        print(f"Error: Invalid index range. START_INDEX ({START_INDEX}) and END_INDEX ({END_INDEX}) are out of bounds for the dataset size ({len(video_ids)}).")
        exit()

    video_ids_to_process = video_ids[START_INDEX:END_INDEX]
    print(f"Total video entries available: {len(video_ids)}")
    print(f"Processing videos from index {START_INDEX} to {END_INDEX - 1}. Total: {len(video_ids_to_process)} videos.")
    print(f"Selected Perturbation Type: {PERTURBATION_TYPE}")

    initial_dataset = datasets.Dataset.from_dict({"video": video_ids_to_process})

    processed_dataset = initial_dataset.map(
        function=make_map_fn(SPLIT_NAME, VIDEO_BASE_DIR, fps_data, PERTURBATION_TYPE, all_video_ids=video_ids),
        batched=True,
        batch_size=1,
        num_proc=max(os.cpu_count() // 2, 1),
        remove_columns=initial_dataset.column_names,
    )

    local_save_dir = os.path.expanduser(LOCAL_SAVE_DIR)
    os.makedirs(local_save_dir, exist_ok=True)
    output_file = os.path.join(local_save_dir, f"{NAME}.parquet")
    processed_dataset.to_parquet(output_file)

    print(f"\n- Processing complete. {len(processed_dataset)} valid samples were generated.")
    print(f"- Dataset saved to: {output_file}")