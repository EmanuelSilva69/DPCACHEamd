import torch
import torch.distributed as dist
import bisect
import pickle
import time
import gc
from collections import deque
from typing import Dict, Any, Optional, List, Callable
from .schedule_utils import derivative_approximation, taylor_formula, precompute_schedule, cal_type
from .cali_utils import calc_cost_matrix_v2


MODEL_BLOCK_CONFIG = {
    "flux": {
        "get_blocks": lambda self, stream: self.single_transformer_blocks if stream == "single_stream" else self.transformer_blocks,
        "cache_stream_list": ["double_stream", "single_stream"],
        "cali_stream": "single_stream",   # stream used during calibration (must be in cache_stream_list)
        "clip_fp16": True,
    },
    "wan": {
        "get_blocks": lambda self, stream: self.blocks,
        "cache_stream_list": ["cond_stream", "uncond_stream"],
        "cali_stream": "cond_stream", 
    },
}


def init_cache(
    self,
    mode="Taylor-DP",
    model_name="flux",
    num_steps=50,
    first_full_steps=3,
    last_full_steps=0,
    order=2,
    selected_steps=None,
    k=None,
    cali=False,
    cost_matrix_path="final_cost_matrix.pkl",
    cache_stream_list=None,
    cache_module_list=["block_output_hidden_states"],
):
    if mode not in ("Taylor-DP",):
        raise ValueError(f"Unsupported mode: {mode}")

    if model_name not in MODEL_BLOCK_CONFIG:
        raise ValueError(f"Unsupported model: {model_name}. Available models: {list(MODEL_BLOCK_CONFIG.keys())}")
    
    if order < 1:
        raise ValueError(f"Order must be >= 1, got {order}")
    
    if first_full_steps < 0 or last_full_steps < 0:
        raise ValueError("first_full_steps and last_full_steps must be non-negative")

    if not cache_stream_list:
        cache_stream_list = MODEL_BLOCK_CONFIG[model_name]["cache_stream_list"]

    empty_cache = {}
    for stream in cache_stream_list:
        empty_cache[stream] = {}
        blocks = MODEL_BLOCK_CONFIG[model_name]["get_blocks"](self, stream)
        for layer in range(len(blocks)):
            empty_cache[stream][layer] = {}
            for module in cache_module_list:
                empty_cache[stream][layer][module] = {}

    cache_data = {
        # basic config
        "mode": mode,
        "model_name": model_name,
        "model_config": MODEL_BLOCK_CONFIG[model_name],
        "order": order,
        "first_full_steps": first_full_steps,
        "last_full_steps": last_full_steps,
        "num_steps": num_steps,
        "cache_stream_list": cache_stream_list,
        "cache_module_list": cache_module_list,
        "cali": cali,
        "selected_steps": selected_steps if selected_steps else [0],
        # cache data
        "cache": empty_cache if mode in ("Taylor-DP",) else None,
        "history": None,
        # runtime state
        "current": {"step": 0, "stream": None, "layer": None, "module": None, "type": None},
    }

    if cali:
        init_history(cache_data, order, cali)

    if selected_steps:
        cache_data["selected_steps"] = selected_steps
    elif mode in ("Taylor-DP") and k:
        cache_data["selected_steps"] = precompute_schedule(
            num_steps,
            k,
            first_full_steps,
            last_full_steps,
            cali=cali,
            cost_matrix_path=cost_matrix_path,
        )
    print(f"Init selected_steps: {cache_data['selected_steps']}")

    return cache_data


def init_history(cache_data, order, cali=False):
    cache_data["history"] = {}
    for stream in cache_data["cache_stream_list"]:
        cache_data["history"][stream] = {}
        for module in cache_data["cache_module_list"]:
            cache_data["history"][stream][module] = deque(maxlen=order + 1) if not cali else deque(maxlen=cache_data["num_steps"] + 1)


def clear_cache(cache_data):
    if not cache_data.get("history"):
        return
    for stream, stream_cache in cache_data["history"].items():
        for module, module_cache in stream_cache.items():
            module_cache.clear()


class CacheHelper:

    def __init__(self, cache_data: Dict[str, Any]):
        self.cache_data = cache_data
        self.current = cache_data["current"]

    # ===== State management =====

    def update_step(self, transformer, timestep: torch.Tensor, reverse=True) -> int:
        """Update diffusion step counter"""
        if transformer is None:
            raise RuntimeError("Please pass a transformer object")

        if not hasattr(transformer, "_step_counter"):
            transformer._step_counter = 0
            transformer._last_timestep_value = None

        current_timestep_value = float(timestep.item())

        if transformer._last_timestep_value is not None and (
            (current_timestep_value > transformer._last_timestep_value + 0.1)
            if reverse
            else (current_timestep_value < transformer._last_timestep_value - 0.1)
        ):
            transformer._step_counter = 0

        if transformer._last_timestep_value != current_timestep_value:
            transformer._step_counter += 1
            transformer._last_timestep_value = current_timestep_value

        return transformer._step_counter - 1
    
    def init_current(self, transformer, timestep: torch.Tensor, reverse: bool=True) -> Dict[str, Any]:
        current_step = self.update_step(transformer, timestep, reverse)
        current = self.cache_data["current"]
        current["step"] = current_step
        if current["step"] == 0:
            clear_cache(self.cache_data)
        cal_type(self.cache_data)
        return current

    def update_stream(self, stream: str, num_layers: int):
        """Update current stream context"""
        self.current["stream"] = stream
        self.current["num_layers"] = num_layers

    def update_layer(self, layer: int):
        """Update current layer index"""
        self.current["layer"] = layer

    def should_compute(self) -> bool:
        """Return True when full forward pass should be computed"""
        return self.current["type"] == "full" or self.cache_data["cali"] or self.current["stream"] not in self.cache_data["cache_stream_list"]

    def should_skip_with_cache(self) -> bool:
        """Return True when cached feature should be reused"""
        return self.current["type"] == "Taylor-DP" and not self.cache_data["cali"] and self.current["stream"] in self.cache_data["cache_stream_list"]

    def is_last_layer(self) -> bool:
        """Return True if current layer is the last layer of current stream"""
        return self.current["layer"] == self.current["num_layers"] - 1

    def should_cache_feature(self) -> bool:
        """Return True when feature should be cached during inference"""
        return not self.cache_data["cali"] and self.is_last_layer() and self.current["stream"] in self.cache_data["cache_stream_list"]

    def should_save_history(self) -> bool:
        """Return True when feature history should be saved during calibration"""
        return self.cache_data["cali"] and self.is_last_layer() and self.current["stream"] in self.cache_data["model_config"]["cali_stream"]

    def print_schedule_status(self, action: str = "compute"):
        if (not dist.is_initialized() or dist.get_rank() == 0) and self.current["layer"] == 0:
            print(f"{action} {'' if action == 'compute' else 'Taylor-DP '}{self.current['stream']} step: {self.current['step']}")

    # ===== Cache operations =====

    def cache_feature(self, feature: torch.Tensor, module_name: str = "block_output_hidden_states"):
        """Cache derivative features up to specified order for current module"""
        self.current["module"] = module_name
        if self.cache_data["mode"] == "Taylor-DP":
            derivative_approximation(cache_data=self.cache_data, feature=feature)

    def save_feature_history(
        self, feature: torch.Tensor, module_name: str = "block_output_hidden_states", preprocess_fn: Optional[Callable] = None,
    ):
        """In calibration mode, save features into history"""
        self.current["module"] = module_name
        cache_tensor = feature.detach()

        if self.cache_data["cali"]:
            cache_tensor = self._gather_if_distributed(cache_tensor)

        if preprocess_fn is not None and not self.cache_data["cali"]:
            cache_tensor = preprocess_fn(cache_tensor)

        if self.cache_data["model_config"].get("clip_fp16", False):
            cache_tensor = CacheHelper.clip_fp16(cache_tensor)

        history = self.cache_data["history"][self.current["stream"]][module_name]
        history.append([cache_tensor, self.current["step"]])

    def retrieve_cached_feature(self, module_name: str = "block_output_hidden_states") -> torch.Tensor:
        """Restore feature from Taylor cache"""
        current_selected_steps = self.cache_data["selected_steps"]
        reference_step = current_selected_steps[bisect.bisect_left(current_selected_steps, self.current["step"]) - 1]

        derivatives_dict = self.cache_data["cache"][self.current["stream"]][self.current["layer"]][module_name]
        distance = self.current["step"] - reference_step

        return taylor_formula(derivatives_dict, distance, self.cache_data["model_config"].get("clip_fp16", False))

    # ===== Calibration operations =====

    def should_perform_calibration(self, cali_stream: str = "single_stream") -> bool:
        """Return True when calibration should be performed on given stream"""
        return (
            self.cache_data["cali"]
            and self.current["stream"] == cali_stream
            and self.current["step"] == self.cache_data["num_steps"] - 1
            and self.cache_data["mode"] in ("Taylor-DP")
            and (not dist.is_initialized() or dist.get_rank() == 0)
        )

    def generate_sentinel_feature(self, cali_stream: str = "single_stream", module_name: Optional[str] = None) -> bool:
        """Generate mock sentinel"""
        if module_name is None:
            module_name = self.cache_data["cache_module_list"][0]

        history = self.cache_data["history"][cali_stream][module_name]
        history_len = len(history)
        max_k = min(self.cache_data["order"] + 1, history_len)

        if max_k > 0:
            tail_feats = [history[-k][0] for k in range(1, max_k + 1)]
            if tail_feats:
                sentinel_feat = torch.stack(tail_feats).mean(0) + 1e-8 * torch.randn_like(tail_feats[0])
                history.append([sentinel_feat.detach(), self.current["step"] + 1])
                return True

        return False

    def compute_and_save_cost_matrix(self, cali_prefix: str = "flux_dpcache", alpha_list: List[float] = None, save_dir: str = ".", cost_metric: str = "l1"):
        """Compute and save cost matrix"""
        if alpha_list is None:
            alpha_list = [0.8]

        cost_matrix_list = calc_cost_matrix_v2(self.cache_data, alpha_list=alpha_list, cost_metric=cost_metric)

        for alpha, cost_matrix in zip(alpha_list, cost_matrix_list):
            if len(alpha_list) > 1:
                cost_file_name = f"cost_matrix_{cali_prefix}_{self.cache_data['order']}_{alpha}.pkl"
            else:
                cost_file_name = f"cost_matrix_{cali_prefix}_{self.cache_data['order']}.pkl"

            file_path = f"{save_dir}/{cost_file_name}" if save_dir != "." else cost_file_name

            with open(file_path, "ab") as f:
                pickle.dump(cost_matrix, f)

            print(f"Saved cost matrix to: {file_path}")
        
        clear_cache(self.cache_data)
        gc.collect()
        torch.cuda.empty_cache()
        del cost_matrix_list

    def perform_calibration(
        self,
        cali_stream: str = "single_stream",
        cali_prefix: str = "flux_dpcache",
        alpha_list: List[float] = None,
        module_name: Optional[str] = None,
        save_dir: str = ".",
        cost_metric: str = "l1",
    ):
        """Full calibration: generate sentinel, compute cost matrix, and save"""
        if self.generate_sentinel_feature(cali_stream, module_name):
            start_time = time.time()
            self.compute_and_save_cost_matrix(cali_prefix, alpha_list, save_dir, cost_metric)
            print(f"Calibration completed. Time taken: {time.time() - start_time:.2f}s")
            clear_cache(self.cache_data)
            torch.cuda.empty_cache()
        else:
            print("Warning: Failed to generate sentinel feature. Calibration skipped.")

    @staticmethod
    def clip_fp16(tensor: torch.Tensor) -> torch.Tensor:
        """Clip fp16 values to avoid overflow"""
        if tensor.dtype == torch.float16:
            return tensor.clip(-65504, 65504)
        return tensor

    def _gather_if_distributed(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather features when running in distributed mode"""
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return tensor
        
        try:
            from xfuser.core.distributed import get_sp_group, get_sequence_parallel_world_size
            if get_sequence_parallel_world_size() > 1:
                return get_sp_group().all_gather(tensor.contiguous(), dim=1)
            return tensor
        except (ImportError, AttributeError, RuntimeError):
            gathered_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_tensors, tensor.contiguous())
            return torch.cat(gathered_tensors, dim=1)
