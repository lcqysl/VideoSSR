import os
import json
import re
import torch
import torch.multiprocessing as mp
import argparse
import numpy as np
import random
from collections import defaultdict
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import threading
import time
import cv2  

CONFIG = {
    "MODEL_PATH": "path/to/model",
    "TARGET_JSON_PATH": "path/to/json",
    "BASE_VIDEO_DIR": "path/to/videos",
    "nframes":32,
    "max_pixels":512**2,
    "DEFAULT_GPUS": 8,
    "max_tokens": 128,
    "PROMPT_TEMPLATE": """{question}\nAnswer with the option letter directly."""
}

    # {
    #     "video": "fFjv93ACGo8",
    #     "question": "...",
    #     "answer": "C"
    # },

def load_model_and_processor(model_path, device):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto" 
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def parse_answer(model_output):
    return model_output[0]

def parse_question(qa_string):
    return qa_string

def process_single_video(qa_item, model, processor, device):
    video_id = qa_item.get("video")
    question = qa_item.get("question")
    correct_answer = qa_item.get("answer")

    video_path = os.path.join(CONFIG["BASE_VIDEO_DIR"], f"{video_id}.mp4")

    if not os.path.exists(video_path):
        print(f"Warning: Video file not found, skipping -> {video_path}")
        return ("video_not_found")

    full_prompt = CONFIG["PROMPT_TEMPLATE"].format(question=question)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video", 
                    "video": f"file://{video_path}", 
                    "fps":2,
                    "max_frames": CONFIG['nframes'], 
                    "max_pixels": CONFIG['max_pixels'],
                    "min_pixels": 128*128
                },
                {
                    "type": "text", 
                    "text": full_prompt
                },
            ],
        }
    ]

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=CONFIG["max_tokens"],do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(output_text)
        predicted_answer = parse_answer(output_text)

        print(f"Video: {video_id} | Predicted: {predicted_answer} | Correct: {correct_answer}")

        if predicted_answer is None:
            return ("invalid_answer")
        elif predicted_answer == correct_answer.strip().upper():
            return ("correct")
        else:
            return ("wrong")

    except Exception as e:
        print(f"Error processing video {video_id}: {e}")
        return ("error")


def aggregate_and_print_results(all_results, total_dataset_size):
    if not all_results:
        print("No results to aggregate.")
        return

    stats = {
        'total': defaultdict(int),
    }

    for result_status in all_results:
        stats['total'][result_status] += 1

    def print_subset_report(subset_name, subset_stats, total_items_in_subset):
        correct = subset_stats.get('correct', 0)
        wrong = subset_stats.get('wrong', 0)
        invalid = subset_stats.get('invalid_answer', 0) 
        error = subset_stats.get('error', 0)
        not_found = subset_stats.get('video_not_found', 0)
        
        processed = correct + wrong

        print(f"--- {subset_name.upper()} SUBSET RESULTS ---")
        print(f"Items in Subset: {total_items_in_subset}")
        print(f"Items Skipped (Video Not Found): {not_found}")
        print(f"Items with Processing Errors: {error}")
        print(f"Items with Invalid Answers: {invalid}") 
        print(f"Total Items for Accuracy Calc: {processed}\n")

        if processed > 0:
            accuracy = (correct / processed) * 100
            print(f"Correct Answers: {correct}")
            print(f"Wrong Answers: {wrong}")
            print(f"Accuracy: {accuracy:.2f}%")
        else:
            print("No items were successfully processed for accuracy calculation.")
        print("-" * 50)

    print("\n" + "="*50)
    print(" " * 16 + "EVALUATION RESULTS")
    print("="*50)

    print_subset_report("Total", stats['total'], total_dataset_size)


def worker(rank, task_queue, results_list, total_dataset_size, batch_size=5):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    device = torch.device(f"cuda:{str(rank)}")

    model, processor = load_model_and_processor(CONFIG["MODEL_PATH"], device)

    while True:
        batch = []
        for _ in range(batch_size):
            try:
                item = task_queue.get_nowait()
                batch.append(item)
            except Exception:
                break

        if not batch:  
            break

        for (original_idx, qa_item) in batch:
            result = process_single_video(qa_item, model, processor, device)
            results_list.append(result)


def progress_monitor(results_list, total_size, stop_event):
    with tqdm(total=total_size, desc="Processing Videos", unit="video") as pbar:
        while not stop_event.is_set():
            current_count = len(results_list)
            pbar.update(current_count - pbar.n) 
            if current_count >= total_size:
                break
            time.sleep(1) 
        pbar.update(total_size - pbar.n)


def main():
    world_size = CONFIG['DEFAULT_GPUS']

    try:
        with open(CONFIG["TARGET_JSON_PATH"], 'r', encoding='utf-8') as f:
            full_dataset = json.load(f)
        total_dataset_size = len(full_dataset)
        print(f"Successfully loaded {total_dataset_size} items from {CONFIG['TARGET_JSON_PATH']}.")
    except Exception as e:
        print(f"Fatal: Error loading dataset. {e}")
        return

    dataset_with_indices = list(enumerate(full_dataset))
    random.shuffle(dataset_with_indices)

    with mp.Manager() as manager:
        task_queue = manager.Queue()
        for item in dataset_with_indices:
            task_queue.put(item)

        results_list = manager.list()

        stop_event = threading.Event()
        progress_thread = threading.Thread(
            target=progress_monitor,
            args=(results_list, total_dataset_size, stop_event)
        )
        progress_thread.start()

        mp.spawn(worker,
                args=(task_queue, results_list, total_dataset_size),
                nprocs=world_size,
                join=True)

        stop_event.set()
        progress_thread.join()

        final_results = list(results_list)

    aggregate_and_print_results(final_results, total_dataset_size)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()