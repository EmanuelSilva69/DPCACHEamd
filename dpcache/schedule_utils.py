import numpy as np
import torch
import pickle
import math
import bisect
from typing import Dict


def compute_derivatives_from_history(history_obj, order):
    if len(history_obj) < 1:
        return {}

    if len(history_obj) < 2:
        return {0: history_obj[-1][0]}

    cache = {}

    def compute_derivative_recursive(idx, order):
        cache_key = (idx, order)
        if cache_key in cache:
            return cache[cache_key]

        cur_tensor, cur_step = history_obj[idx]
        if order == 0:
            result = cur_tensor
        elif idx == 0:
            result = None
        else:
            prev_step = history_obj[idx - 1][1]

            current_deriv = compute_derivative_recursive(idx, order - 1)
            prev_deriv = compute_derivative_recursive(idx - 1, order - 1)

            if current_deriv is None or prev_deriv is None:
                return None

            step_distance = cur_step - prev_step
            result = (current_deriv - prev_deriv) / step_distance

        cache[cache_key] = result
        return result

    try:
        derivatives_dict = {}
        for order in range(order + 1):
            deriv = compute_derivative_recursive(len(history_obj) - 1, order)
            if deriv is None:
                break
            derivatives_dict[order] = deriv

        return derivatives_dict
    finally:
        cache.clear()
        del cache


def precompute_schedule(
    total_steps: int,
    k: int,
    first_full_steps: int = 3,
    last_full_steps: int = 0,
    cali=False,
    cost_matrix_path: str = "final_cost_matrix.pkl",
) -> list:
    if not (first_full_steps >= 0 and last_full_steps >= 0):
        raise ValueError("Invalid input parameters.")

    extended_total_steps = total_steps + 1

    mandatory_steps = set(range(first_full_steps))
    if last_full_steps > 0:
        mandatory_steps.update(range(total_steps - last_full_steps, total_steps))
    mandatory_steps.add(0)
    mandatory_steps.add(total_steps)

    effective_k = k + 1
    if effective_k > extended_total_steps:
        effective_k = extended_total_steps

    is_mandatory = np.zeros(extended_total_steps, dtype=bool)
    for m in mandatory_steps:
        is_mandatory[m] = True

    if not cali:
        with open(cost_matrix_path, "rb") as f:
            cost_data = pickle.load(f)

        if cost_data.ndim != 3:
            raise ValueError(f"Expected a 3D cost tensor, but got {cost_data.ndim} dimensions.")

        # adjust cost_matrix according to mandatory
        for j in range(extended_total_steps):
            for i in range(j + 1, extended_total_steps):
                if np.any(is_mandatory[j + 1 : i]):
                    cost_data[:, j, i] = np.inf

        expected_step_distance = round((total_steps - first_full_steps) / (k - first_full_steps)) if k > first_full_steps else 1
        print(f"Using 3D cost tensor with Dynamic K-Selection. Expected step distance: {expected_step_distance}")
        cost_tensor_3d = cost_data
    else:
        return list(range(total_steps))

    dp = np.full((effective_k, extended_total_steps), np.inf)  # dp[k, i]: min cost to reach step i with k jumps
    path = np.zeros((effective_k, extended_total_steps), dtype=int)  # best last jump index for reaching step i with k jumps
    dp[0, 0] = 0
    max_jump_step = max(int(total_steps * 0.3), 1)

    for k in range(1, effective_k):
        for i in range(k, extended_total_steps):
            min_cost, best_j = np.inf, -1
            search_start_j = max(k - 1, i - max_jump_step)
            for j in range(search_start_j, i):
                if dp[k - 1, j] != np.inf:
                    transition_cost = np.inf
                    if j == 0:
                        transition_cost = cost_tensor_3d[0, 0, i]
                    elif j > 0 and is_mandatory[j] and is_mandatory[j - 1]:
                        transition_cost = cost_tensor_3d[j - 1, j, i]
                    else:
                        transition_cost = cost_tensor_3d[path[k - 1, j], j, i]

                    if transition_cost != np.inf:
                        current_total_cost = dp[k - 1, j] + transition_cost
                        if current_total_cost < min_cost:
                            min_cost, best_j = current_total_cost, j

            if best_j != -1:
                dp[k, i], path[k, i] = min_cost, best_j

    best_end_step = total_steps

    if np.isinf(dp[effective_k - 1, best_end_step]):
        print("Sentinel was unreachable. Finding best available path.")
        best_end_step = np.nanargmin(dp[effective_k - 1, :])

    schedule_with_sentinel = [0] * effective_k
    schedule_with_sentinel[-1] = best_end_step

    current_step = best_end_step
    for k in range(effective_k - 1, 0, -1):
        prev_step = path[k, current_step]
        schedule_with_sentinel[k - 1], current_step = prev_step, prev_step

    return schedule_with_sentinel[:-1]


def derivative_approximation(cache_data: Dict, feature: torch.Tensor):
    current_selected_steps = cache_data["selected_steps"]
    current = cache_data["current"]
    last_pos = bisect.bisect_left(current_selected_steps, current["step"])
    difference_distance = current["step"] - current_selected_steps[max(last_pos - 1, 0)]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_data["order"]):
        if (cache_data["cache"][current["stream"]][current["layer"]][current["module"]].get(i, None) is not None) and (
            current["step"] > cache_data["first_full_steps"] - 2
        ):
            updated_taylor_factors[i + 1] = (
                updated_taylor_factors[i] - cache_data["cache"][current["stream"]][current["layer"]][current["module"]][i]
            ) / difference_distance
        else:
            break

    cache_data["cache"][current["stream"]][current["layer"]][current["module"]] = updated_taylor_factors


def taylor_formula(derivative_dict: Dict, distance: int, clip_fp16=False) -> torch.Tensor:
    output = 0
    for i in range(len(derivative_dict)):
        if clip_fp16:
            output += (1 / math.factorial(i)) * derivative_dict[i].to(torch.float32) * (distance**i)
        else:
            output += (1 / math.factorial(i)) * derivative_dict[i] * (distance**i)
    
    return output.to(torch.float16) if clip_fp16 else output


def cal_type(cache_data):
    current = cache_data["current"]
    if current["step"] in cache_data["selected_steps"]:
        current["type"] = "full"
    elif cache_data["mode"]:
        current["type"] = cache_data["mode"]
    else:
        raise ValueError("Invalid calculation type")
