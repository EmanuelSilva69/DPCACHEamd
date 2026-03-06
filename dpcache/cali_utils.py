from collections import deque
from .schedule_utils import compute_derivatives_from_history, taylor_formula
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import os
from pathlib import Path


def compute_cost(x_pred: torch.Tensor, x_gt: torch.Tensor, metric: str = "l1", clip_fp16: bool = True) -> float:
    x_pred_clip = x_pred.clip(-65504, 65504) if clip_fp16 else x_pred
    x_gt_clip = x_gt.clip(-65504, 65504) if clip_fp16 else x_gt
    
    if metric == "l1":
        diff = x_pred_clip / 2 - x_gt_clip / 2
        return diff.abs().mean().item()
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def calc_cost_matrix_v2(cache_data, alpha_list=[0.8], cost_metric="l1"):
    stream = cache_data["current"]["stream"]
    module = cache_data["cache_module_list"][0]
    mode = cache_data["mode"]
    num_steps = cache_data["num_steps"]
    history_obj = cache_data["history"][stream][module]
    clip_fp16 = cache_data["model_config"].get("clip_fp16", False)
    device = history_obj[0][0].device
    order = cache_data["order"]
    history_steps = [item[1] for item in history_obj]
    history_tensors = [item[0] for item in history_obj]

    reference_step = num_steps - 1
    step_derivatives = compute_derivatives_from_history(deque([history_obj[m] for m in range(num_steps - 1 - order, num_steps)]), order)
    sentinel_pred = taylor_formula(step_derivatives, num_steps - reference_step, clip_fp16)

    smoothed_sentinel_pred_list = [alpha * sentinel_pred + (1 - alpha) * history_obj[num_steps][0] for alpha in alpha_list]
    cost_matrix_list = []

    print(f"Calculating 3d cost tensor using {mode} with {cost_metric} metric.")
    anchor_dist_limit = max(int(num_steps * 0.3), 1)

    for alpha_idx, alpha in enumerate(alpha_list):
        cost_tensor_3d = torch.full((num_steps, num_steps + 1, num_steps + 1), float("inf"), device=device)
        current_sentinel = smoothed_sentinel_pred_list[alpha_idx]

        for i in range(order):
            cost_tensor_3d[: i + 1, i, i + 1] = 0

        for i in range(order, num_steps):
            for anchor in range(max(0, i - anchor_dist_limit), i):
                if anchor < order - 1:
                    continue
                accumulated_error = 0.0

                reference_step = history_steps[i]

                step_i_derivatives = compute_derivatives_from_history(
                    deque([history_obj[m] for m in range(anchor - (order - 1), anchor + 1)] + [history_obj[i]]), order
                )

                for j in range(i + 1, num_steps + 1):
                    distance = j - reference_step
                    x_pred = taylor_formula(step_i_derivatives, distance, clip_fp16)
                    x_gt = current_sentinel if j == num_steps else history_tensors[j]
                    cost = compute_cost(x_pred, x_gt, metric=cost_metric, clip_fp16=clip_fp16)
                    accumulated_error += cost
                    cost_tensor_3d[anchor, i, j] = accumulated_error

        cost_matrix_list.append(cost_tensor_3d.detach().cpu().numpy())

    del history_tensors, history_steps, step_derivatives, sentinel_pred, smoothed_sentinel_pred_list
    return cost_matrix_list


def merge_cost_matrix(pkl_path="cost_matrix.pkl", save_path=None):
    try:
        with open(pkl_path, "rb") as f:
            data_list = []
            while True:
                try:
                    data_list.append(pickle.load(f))
                except EOFError:
                    break

        if not data_list:
            print("No data found in pkl file.")
            return None

        first_item_ndim = data_list[0].ndim
        print(f"Merging data with {first_item_ndim} dimensions.")

        stack = np.array(data_list)

        valid = ~np.isinf(stack)
        counts = np.sum(valid, axis=0)
        sums = np.nansum(np.where(valid, stack, 0), axis=0)

        mean_data = np.full(stack.shape[1:], np.inf)
        mean_data[counts > 0] = sums[counts > 0] / counts[counts > 0]

        if save_path is None:
            mode_str = "3d" if first_item_ndim == 3 else "2d"
            save_path = f"final_{mode_str}_{Path(pkl_path).stem}.pkl"

        with open(save_path, "wb") as f:
            pickle.dump(mean_data, f)

        return mean_data

    except FileNotFoundError:
        print(f"{pkl_path} not found.")
        return None
    except Exception as e:
        print(e)
        return None


def merge_calibration_results(cali_prefix, order=2):
    pkl_path = f"cost_matrix_{cali_prefix}_{order}.pkl"
    save_path = f"final_cost_matrix_{cali_prefix}.pkl"

    print("Merging calibration results...")
    result = merge_cost_matrix(pkl_path=pkl_path, save_path=save_path)

    if result is not None:
        os.remove(pkl_path)
        print(f"Successfully merged cost matrix to {save_path}")
        return save_path
    else:
        print("Failed to merge cost matrix")
        return None


if __name__ == "__main__":
    result = merge_cost_matrix(pkl_path="cost_matrix_flux.pkl")
    print(result)
