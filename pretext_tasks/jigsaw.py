import os
import datasets
import random
import json

SYSTEM_PROMPT = """Based on this video, provide the answer directly."""
NUM_CLIPS = 6
TOTAL_FRAMES_NEEDED = 48
FRAMES_PER_CLIP = TOTAL_FRAMES_NEEDED // NUM_CLIPS
if NUM_CLIPS==6:    
    QUESTION_TEXT = "This video is presented as 6 separate clips, which are in a shuffled order. Your task is to determine the correct chronological sequence. Please output a six-digit number that specifies the order in which to play the clips you are seeing (labeled 1 through 6 by their position). For example, if you decide that the clip at position 3 of this video is the true beginning (the 1st clip of the original video), the clip at position 4 is the 2nd part, the clip at position 1 is the 3rd part, the clip at position 5 is the 4th part, the clip at position 6 is the 5th part, and finally the clip at position 2 is the 6th part, then your answer should be '341562'."
elif NUM_CLIPS==8:    
    QUESTION_TEXT = "This video is presented as 8 separate clips, which are in a shuffled order. Your task is to determine the correct chronological sequence. Please output an eight-digit number that specifies the order in which to play the clips you are seeing (labeled 1 through 8 by their position). For example, if you decide that the clip at position 5 of this video is the true beginning (the 1st clip of the original video), the clip at position 1 is the 2nd part, the clip at position 8 is the 3rd part, the clip at position 2 is the 4th part, the clip at position 7 is the 5th part, the clip at position 4 is the 6th part, the clip at position 6 is the 7th part, and finally the clip at position 3 is the 8th part, then your answer should be '51827463'."
SPLIT_NAME = "train"
START_INDEX = 12000
END_INDEX = 24000 

VIDEO_BASE_DIR = f"path/to/{TOTAL_FRAMES_NEEDED}frames"
FPS_JSON_PATH = f"path/to/{TOTAL_FRAMES_NEEDED}frames_fps.json"
LOCAL_SAVE_DIR = "path/to/parquet"
NAME = f"3jigsaw{NUM_CLIPS}"
MAX_PIXELS = 256

NUM_SHUFFLES_PER_VIDEO = 1

def make_map_fn(split, video_base_dir, fps_data, num_shuffles_per_video):
    def process_fn(batch):
        results = []
        for idx, example in enumerate(batch["video"]):
            video_id = example
            video_path = os.path.join(video_base_dir, video_id)
            sampling_fps = fps_data.get(video_id)
            if sampling_fps is None:
                print(f"[Warning] Skipping video ID '{video_id}': No FPS data found in the JSON file.")
                continue

            if not os.path.isdir(video_path):
                print(f"[Warning] Skipping video ID '{video_id}': The directory was not found at the expected path '{video_path}'.")
                continue

            try:
                frames = sorted(
                    [f for f in os.listdir(video_path) if f.endswith(".jpg") and "_" not in f],
                    key=lambda x: int(x.split(".")[0]),
                )
            except ValueError:
                continue

            if len(frames) < TOTAL_FRAMES_NEEDED:
                continue

            groups = [
                frames[i : i + FRAMES_PER_CLIP]
                for i in range(0, TOTAL_FRAMES_NEEDED, FRAMES_PER_CLIP)
            ]

            for shuffle_idx in range(num_shuffles_per_video):
                shuffled_appearance_order = list(range(NUM_CLIPS))
                random.shuffle(shuffled_appearance_order)

                shuffled_groups = [groups[i] for i in shuffled_appearance_order]

                answer_list = [0] * NUM_CLIPS
                for position_idx, original_identity in enumerate(shuffled_appearance_order):
                    answer_list[original_identity] = str(position_idx + 1)
                answer = "".join(answer_list)

                prompt_clips_text = "\n".join([f"Clip {i+1}: <video>" for i in range(NUM_CLIPS)])
                full_user_prompt = f"{prompt_clips_text}\n{QUESTION_TEXT}"

                videos_list = []
                for group in shuffled_groups:
                    frame_paths_for_clip = [os.path.join(video_path, f) for f in group]
                    clip_data = {
                        "video": frame_paths_for_clip,
                        "sample_fps": sampling_fps,
                        "max_pixels": MAX_PIXELS * MAX_PIXELS,
                        "min_pixels": 128 * 128,
                    }
                    videos_list.append(clip_data)

                results.append(
                    {
                        "data_source": "shuffled",
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": full_user_prompt},
                        ],
                        "videos": videos_list,
                        "ability": "video_understanding",
                        "reward_model": {"style": "rule", "ground_truth": answer},
                        "extra_info": {
                            "split": split,
                            "index": idx,
                            "shuffle_id": shuffle_idx,
                            "answer": answer,
                            "question": QUESTION_TEXT,
                            "video_id": video_id,
                            "sampling_fps": sampling_fps,
                        },
                    }
                )

        if not results:
            return {}
        output = {k: [item[k] for item in results] for k in results[0].keys()}
        return output

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
    
    initial_dataset = datasets.Dataset.from_dict({"video": video_ids_to_process})


    processed_dataset = initial_dataset.map(
        function=make_map_fn(SPLIT_NAME, VIDEO_BASE_DIR, fps_data, NUM_SHUFFLES_PER_VIDEO),
        batched=True,
        batch_size=1,
        num_proc=max(os.cpu_count() // 2, 1),
        remove_columns=initial_dataset.column_names,
    )

    local_save_dir = os.path.expanduser(LOCAL_SAVE_DIR)
    os.makedirs(local_save_dir, exist_ok=True)
    output_file = os.path.join(local_save_dir, f"{NAME}.parquet")
    processed_dataset.to_parquet(output_file)

    print(f"\n- Processing complete. {len(processed_dataset)} valid shuffled samples were generated.")
    print(f"- Dataset saved to: {output_file}")