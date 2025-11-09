import re
from typing import Optional, Tuple

def print_answer(predict_str: str, ground_truth: str, type) -> None:
    print("\n----------------------------------------------------")
    print(f"Prediction: {predict_str}")
    print(f"Ground Truth: {type} {ground_truth}")
    print("----------------------------------------------------\n")

def parse_interval(s: str) -> Optional[Tuple[float, float]]:
    if not isinstance(s, str):
        return None
        
    s = s.strip()
    
    match = re.match(r"^\s*([\d\.]+)\s*-\s*([\d\.]+)\s*$", s)
    if not match:
        return None
        
    try:
        start = float(match.group(1))
        end = float(match.group(2))
        
        if start > end:
            return None
            
        return start, end
    except (ValueError, TypeError):
        return None

def calculate_iou(pred_interval: Tuple[float, float], gt_interval: Tuple[float, float]) -> float:
    pred_start, pred_end = pred_interval
    gt_start, gt_end = gt_interval

    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    
    intersection_length = max(0, intersection_end - intersection_start)

    pred_length = pred_end - pred_start
    gt_length = gt_end - gt_start
    
    union_length = pred_length + gt_length - intersection_length
    
    epsilon = 1e-9
    if union_length < epsilon:
        return 1.0 if intersection_length > 0 else 0.0

    iou = intersection_length / union_length
    return iou

def compute_score(predict_str: str, ground_truth: str, extra_info: dict) -> float:
    print_answer(predict_str, ground_truth,extra_info["perturbation_type"])

    pred_interval = parse_interval(predict_str)
    gt_interval = parse_interval(ground_truth)
    
    if pred_interval is None or gt_interval is None:
        print("Failed to parse prediction or ground truth. Score: 0.0")
        print("----------------------------------------------------\n")
        return 0.0
        
    iou_score = calculate_iou(pred_interval, gt_interval)
    
    print(f"mIoU Score: {iou_score:.4f}")
    print("----------------------------------------------------\n")
    
    return iou_score