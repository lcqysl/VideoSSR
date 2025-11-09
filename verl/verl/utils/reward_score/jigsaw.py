import re
from typing import Optional

def print_answer(predict_str,ground_truth):
    print("\n----------------------------------------------------")
    print(predict_str)
    print("ground_truth: ",ground_truth)
    print("----------------------------------------------------\n")


def is_valid_permutation(s: str, reference: str) -> bool:
    return sorted(s) == sorted(reference)


def acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = predict_str
    if not answer:
        return 0.0

    clean_answer = answer.strip()
    clean_ground_truth = ground_truth.strip()

    if not is_valid_permutation(clean_answer, clean_ground_truth):
        return 0.0

    total_distance_error = 0
    for char_to_find in clean_ground_truth:
        try:
            pos_pred = clean_answer.find(char_to_find)
            pos_gt = clean_ground_truth.find(char_to_find)
        except ValueError:
            return 0.0

        distance = abs(pos_pred - pos_gt)
        total_distance_error += distance

    n = len(clean_ground_truth)
    if n == 0:
        return 1.0 

    max_possible_error = n * n // 2
    if max_possible_error == 0:
        return 1.0

    normalized_error = total_distance_error / max_possible_error
    reward = 1.0 - normalized_error
    return reward


def compute_score(predict_str: str, ground_truth: str, extra_info: dict) -> float:
    print_answer(predict_str, ground_truth)
    accuracy_component = acc_reward(predict_str, ground_truth)
    return accuracy_component