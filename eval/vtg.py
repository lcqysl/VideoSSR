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
    "DEFAULT_GPUS": 8,
    "max_pixels": 512 ** 2,          
    "nframes": 32,       
    "max_tokens": 128,
    "PROMPT_TEMPLATE": """Please find the visual event described by a sentence in the video, determining its starting and ending times. The format should be: 'The event happens in the start time - end time'. For example, The event 'person turn a light on' happens in the 24.3 - 30.4 seonds. Now I will give you the textual sentence: {question} Please return its start time and end time."""
}



def load_model_and_processor(model_path, device):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto" 
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def parse_timestamp_answer(model_output: str):
    if not model_output:
        return None
    match = re.search(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', model_output.strip())
    
    if match:
        try:
            start_time = float(match.group(1))
            end_time = float(match.group(2))
            return min(start_time, end_time), max(start_time, end_time)
        except (ValueError, IndexError):
            return None
    return None

def calculate_iou(pred_box, gt_box):
    """
    Calculate Intersection over Union (IoU) for two temporal segments.
    Each box is a tuple (start_time, end_time).
    """
    pred_start, pred_end = pred_box
    gt_start, gt_end = gt_box

    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0, inter_end - inter_start)

    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection

    if union == 0:
        return 0.0  

    return intersection / union

def parse_question(qa_string):
    return qa_string

def process_single_video(qa_item, model, processor, device):
    video_id = qa_item.get("video")
    question = qa_item.get("question")
    correct_answer_times = qa_item.get("answer") 

    video_path = os.path.join(CONFIG["BASE_VIDEO_DIR"], f"{video_id}.mp4")

    if not os.path.exists(video_path):
        print(f"Warning: Video file not found, skipping -> {video_path}")
        return {"status": "video_not_found", "iou": None}

    full_prompt = CONFIG["PROMPT_TEMPLATE"].format(question=question)
    messages = [{"role": "user", "content": [{"type": "video", "video": f"file://{video_path}","fps":2 ,"max_frames": CONFIG['nframes'], "max_pixels": CONFIG['max_pixels'], "min_pixels": 128*128}, {"type": "text", "text": full_prompt}]}]

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        inputs = processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs).to(model.device)
        
        generated_ids = model.generate(**inputs, max_new_tokens=CONFIG["max_tokens"])
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(output_text)
        predicted_times = parse_timestamp_answer(output_text)

        if predicted_times is None:
            print(f"error: {output_text}")
            return {"status": "parse_failed", "iou": None}
        
        iou = calculate_iou(predicted_times, tuple(parse_timestamp_answer(correct_answer_times)))
        print(iou)
        return {"status": "success", "iou": iou}

    except Exception as e:
        print(f"Error processing video {video_id}: {e}")
        return {"status": "inference_error", "iou": None}

def aggregate_and_print_results(all_results, total_dataset_size):
    if not all_results:
        print("No results to aggregate.")
        return

    stats = defaultdict(int)
    iou_scores = []

    for result in all_results:
        status = result['status']
        iou = result['iou']
        stats[status] += 1
        if status == "success" and iou is not None:
            iou_scores.append(iou)

    print("\n" + "="*50)
    print(" " * 16 + "EVALUATION RESULTS (mIoU)")
    print("="*50)

    total_processed = len(all_results)
    not_found = stats['video_not_found']
    video_error = stats['video_error']
    inference_error = stats['inference_error']
    parse_failed = stats['parse_failed']
    success = stats['success']

    print(f"Total Items in Dataset: {total_dataset_size}")
    print(f"Total Items Processed: {total_processed}")
    print("-" * 50)
    print(f"Items Skipped (Video Not Found): {not_found}")
    print(f"Items with Video Reading Errors: {video_error}")
    print(f"Items with Inference Errors: {inference_error}")
    print(f"Items with Unparsable Answers: {parse_failed}")
    print(f"Successfully Processed & Parsed: {success}")
    print("-" * 50)

    if iou_scores:
        mean_iou = np.mean(iou_scores)
        print(f"Mean IoU (mIoU): {mean_iou:.4f}")

        for threshold in [0.3, 0.5, 0.7]:
            accuracy_at_threshold = np.mean([1 if iou >= threshold else 0 for iou in iou_scores]) * 100
            print(f"Accuracy @ IoU={threshold}: {accuracy_at_threshold:.2f}%")
    else:
        print("No successful results to calculate mIoU.")
    
    print("="*50)
    print(f"Model Path: {CONFIG['MODEL_PATH']}")
    print("="*50)


def worker(rank, task_queue, results_list, batch_size=5):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    device = torch.device(f"cuda:{str(rank)}")
    model, processor = load_model_and_processor(CONFIG["MODEL_PATH"], device)

    while True:
        batch = []
        for _ in range(batch_size):
            try:
                qa_item = task_queue.get_nowait()
                batch.append(qa_item)
            except Exception:
                break
        
        if not batch:
            break

        for qa_item in batch:
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
    parser = argparse.ArgumentParser(description="Run multi-GPU video QA evaluation for temporal localization.")
    parser.add_argument("-n", "--num_gpus", type=int, default=CONFIG["DEFAULT_GPUS"],
                        help=f"Number of GPUs to use. Default: {CONFIG['DEFAULT_GPUS']}")
    args = parser.parse_args()
    world_size = args.num_gpus

    try:
        with open(CONFIG["TARGET_JSON_PATH"], 'r', encoding='utf-8') as f:
            full_dataset = json.load(f)
        total_dataset_size = len(full_dataset)
        print(f"Successfully loaded {total_dataset_size} items from {CONFIG['TARGET_JSON_PATH']}.")
    except Exception as e:
        print(f"Fatal: Error loading dataset. {e}")
        return
    
    random.shuffle(full_dataset)

    with mp.Manager() as manager:
        task_queue = manager.Queue()
        for item in full_dataset:
            task_queue.put(item)

        results_list = manager.list()

        stop_event = threading.Event()
        progress_thread = threading.Thread(
            target=progress_monitor,
            args=(results_list, total_dataset_size, stop_event)
        )
        progress_thread.start()

        mp.spawn(worker,
                 args=(task_queue, results_list),
                 nprocs=world_size,
                 join=True)

        stop_event.set()
        progress_thread.join()

        final_results = list(results_list)

    aggregate_and_print_results(final_results, total_dataset_size)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()