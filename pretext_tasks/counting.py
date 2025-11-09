import os
import datasets
import random
import json
import cv2  
import numpy as np
from collections import defaultdict
import shutil

SYSTEM_PROMPT = """Based on this video, provide the answer directly."""
QUESTION_TEXT = "Count the number of circles, squares, and triangles that appear in this video. Be aware that the shapes can appear in any color and at any angle of rotation. They may be present on one or multiple frames, and any given frame can contain more than one shape. Provide the answer as three comma-separated numbers in the format: circles,squares,triangles. For example, if you see 3 circles, 1 square, and 4 triangles, your answer should be '3,1,4'."
SHAPES_TO_COUNT = ["circle", "square", "triangle"]
MAX_FRAMES_PER_SHAPE = 3
MAX_INSTANCES_PER_FRAME = 3
MIN_FRAME_INTERVAL = 8

TOTAL_FRAMES_NEEDED = 48
SPLIT_NAME = "train"
START_INDEX = 24000
END_INDEX = 30000
MAX_PIXELS=256

VIDEO_BASE_DIR = f"path/to/{TOTAL_FRAMES_NEEDED}frames"
FPS_JSON_PATH = f"path/to/{TOTAL_FRAMES_NEEDED}frames_fps.json"
LOCAL_SAVE_DIR = "path/to/parquet"
NAME = f"counting"


def draw_circle(image, center, radius, color):
    cv2.circle(image, center, radius, color, -1)

def draw_square(image, center, size, color, angle=0):
    half_size = size // 2
    points = np.array([
        [-half_size, -half_size],
        [ half_size, -half_size],
        [ half_size,  half_size],
        [-half_size,  half_size]
    ], dtype=np.float32)

    M = cv2.getRotationMatrix2D(center=(0, 0), angle=angle, scale=1.0)

    rotated_points = np.dot(points, M[:, :2].T)

    translated_points = rotated_points + center

    cv2.fillPoly(image, [translated_points.astype(np.int32)], color)

def draw_triangle(image, center, size, color, angle=0):
    height = int(size * (3**0.5) / 2)
    half_base = size // 2

    points = np.array([
        [0, -height / 2],          
        [-half_base, height / 2],  
        [half_base, height / 2]    
    ], dtype=np.float32)

    M = cv2.getRotationMatrix2D(center=(0, 0), angle=angle, scale=1.0)
    
    rotated_points = np.dot(points, M[:, :2].T)

    translated_points = rotated_points + center
    
    cv2.drawContours(image, [translated_points.astype(np.int32)], 0, color, -1)



def make_map_fn(split, video_base_dir, fps_data):
    
    def process_fn(batch):
        results = []
        for idx, video_id in enumerate(batch["video"]):
            video_path = os.path.join(video_base_dir, video_id)
            sampling_fps = fps_data.get(video_id)

            if not os.path.isdir(video_path):
                continue

            try:
                frames = sorted(
                    [f for f in os.listdir(video_path) if f.endswith(".jpg") and "_" not in f],
                    key=lambda x: int(x.split(".")[0]),
                )[:TOTAL_FRAMES_NEEDED]
            except (ValueError, IndexError):
                continue

            if len(frames) < TOTAL_FRAMES_NEEDED:
                continue

            first_frame_path = os.path.join(video_path, frames[0])
            first_frame = cv2.imread(first_frame_path)
            if first_frame is None: continue
            h, w, _ = first_frame.shape
            
            
            total_counts = {shape: 0 for shape in SHAPES_TO_COUNT}
            frame_to_shape_info = {}
            
            available_indices = set(range(len(frames)))

            for shape in SHAPES_TO_COUNT:
                if not available_indices: 
                    break
                
                num_frames_to_modify = random.randint(1, MAX_FRAMES_PER_SHAPE)
                
                population = list(available_indices)
                random.shuffle(population)
                
                selected_for_this_shape = []
                for index in population:
                    is_valid = True
                    for selected in selected_for_this_shape:
                        if abs(index - selected) <= MIN_FRAME_INTERVAL:
                            is_valid = False
                            break
                    if is_valid:
                        selected_for_this_shape.append(index)
                    if len(selected_for_this_shape) == num_frames_to_modify:
                        break
                
                if selected_for_this_shape:
                    available_indices.difference_update(selected_for_this_shape)

                    for frame_idx in selected_for_this_shape:
                        num_instances = random.randint(1, MAX_INSTANCES_PER_FRAME)
                        total_counts[shape] += num_instances
                        frame_to_shape_info[frame_idx] = {"type": shape, "count": num_instances}

            final_frame_paths_for_video = []
            for i, frame_name in enumerate(frames):
                if i in frame_to_shape_info:
                    shape_info = frame_to_shape_info[i]
                    shape_type = shape_info["type"]
                    instance_count = shape_info["count"]
                    
                    original_frame_path = os.path.join(video_path, frame_name)
                    image = cv2.imread(original_frame_path)

                    central_ratio = 0.7
                    central_w = int(w * central_ratio)
                    central_h = int(h * central_ratio)
                    offset_x = (w - central_w) // 2
                    offset_y = (h - central_h) // 2

                    generated_positions = []
                    
                    for _ in range(instance_count):
                        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                        min_dim = min(h, w)
                        size = random.randint(int(min_dim * 0.2), int(min_dim * 0.4))
                        
                        max_retries = 20  
                        for _ in range(max_retries):
                            margin = int(size * 0.75) 
                            
                            x_range_start = offset_x + margin
                            x_range_end = offset_x + central_w - margin
                            y_range_start = offset_y + margin
                            y_range_end = offset_y + central_h - margin

                            if x_range_start >= x_range_end or y_range_start >= y_range_end:
                                margin = int(size * 0.75)
                                center_x = random.randint(margin, w - margin)
                                center_y = random.randint(margin, h - margin)
                                break 

                            center_x = random.randint(x_range_start, x_range_end)
                            center_y = random.randint(y_range_start, y_range_end)
                            
                            is_far_enough = True
                            for pos_x, pos_y in generated_positions:
                                distance = ((center_x - pos_x)**2 + (center_y - pos_y)**2)**0.5
                                if distance < size: 
                                    is_far_enough = False
                                    break
                            
                            if is_far_enough:
                                break 
                        
                        generated_positions.append((center_x, center_y))

                        random_angle = random.uniform(0, 360)
                        if shape_type == "circle":
                            draw_circle(image, (center_x, center_y), size // 2, color)
                        elif shape_type == "square":
                            draw_square(image, (center_x, center_y), size, color, random_angle)
                        elif shape_type == "triangle":
                            draw_triangle(image, (center_x, center_y), size, color, random_angle)
                    
                    base_name, ext = os.path.splitext(frame_name)
                    new_frame_name = f"{base_name}_{shape_type}{instance_count}{ext}"
                    new_frame_path = os.path.join(video_path, new_frame_name)
                    cv2.imwrite(new_frame_path, image)
                    
                    final_frame_paths_for_video.append(new_frame_path)
                else:
                    original_frame_path = os.path.join(video_path, frame_name)
                    final_frame_paths_for_video.append(original_frame_path)

            answer = ",".join([str(total_counts[shape]) for shape in SHAPES_TO_COUNT])

            results.append({
                "data_source": "counting",
                "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"<video>\n{QUESTION_TEXT}"}],
                "videos": [{"video": final_frame_paths_for_video, "sample_fps": sampling_fps,"max_pixels": MAX_PIXELS * MAX_PIXELS,"min_pixels": 128 * 128,}],
                "ability": "video_understanding",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": split, "answer": answer, "question": QUESTION_TEXT, "video_id": video_id},
            })

        if not results: return {}
        output = {k: [dic[k] for dic in results] for k in results[0]}
        return output

    return process_fn


if __name__ == "__main__":
    print(f"Loading FPS data from: {FPS_JSON_PATH}")
    with open(FPS_JSON_PATH, "r") as f:
        fps_data = json.load(f)

    print("\nWARNING: This script will write new, modified frame images directly into your source video directories specified in VIDEO_BASE_DIR.\n")

    video_ids = list(fps_data.keys())
    video_ids_to_process = video_ids[START_INDEX:END_INDEX]

    print(f"Processing videos from index {START_INDEX} to {END_INDEX - 1}. Total: {len(video_ids_to_process)} videos.")
    
    initial_dataset = datasets.Dataset.from_dict({"video": video_ids_to_process})

    processed_dataset = initial_dataset.map(
        function=make_map_fn(SPLIT_NAME, VIDEO_BASE_DIR, fps_data),
        batched=True,
        batch_size=1,
        num_proc=max(os.cpu_count() // 2, 1),
        remove_columns=initial_dataset.column_names,
    )

    os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)
    output_file = os.path.join(LOCAL_SAVE_DIR, f"{NAME}.parquet")
    processed_dataset.to_parquet(output_file)

    print(f"\n- Processing complete. {len(processed_dataset)} samples were generated for the counting game.")
    print(f"- Dataset saved to: {output_file}")