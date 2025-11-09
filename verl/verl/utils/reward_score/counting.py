import re
from typing import List, Optional

def calculate_numerical_reward(pred_num: float, gt_num: float) -> float:
    if gt_num == pred_num:
        return 1.0
        
    epsilon = 1e-9 
    relative_error = abs(pred_num - gt_num) / (abs(gt_num) + epsilon)

    reward = max(0.0, 1.0 - relative_error)
    
    return reward

def compute_score(predict_str: str, ground_truth: str, extra_info: dict) -> float:
    print("\n------------------- Evaluation -------------------")
    print(f"Model Prediction: '{predict_str}'")
    print(f"Ground Truth:     '{ground_truth}'")
    print("----------------------------------------------------")

    try:
        gt_numbers = [float(num) for num in ground_truth.split(',')]
    except (ValueError, AttributeError):
        print("Error: Ground truth is malformed. Cannot compute score.")
        return 0.0

    pred_numbers_str = re.findall(r'\d+', predict_str)
    pred_numbers = [float(num) for num in pred_numbers_str]
    
    if len(pred_numbers) != len(gt_numbers):
        print("Final Score: 0.0")
        print("----------------------------------------------------\n")
        return 0.0

    individual_scores = []
    for i in range(len(gt_numbers)):
        gt_num = gt_numbers[i]
        pred_num = pred_numbers[i]
        
        score = calculate_numerical_reward(pred_num, gt_num)
        individual_scores.append(score)

    if not individual_scores: 
        final_score = 0.0
    else:
        final_score = sum(individual_scores) / len(individual_scores)
    
    print("----------------------------------------------------")
    print(f"Final Score: {final_score:.4f}")
    print("----------------------------------------------------\n")

    return final_score

