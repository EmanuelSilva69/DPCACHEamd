import torch
import os
import pandas as pd
from tqdm import tqdm
from diffusers import DiffusionPipeline
from dpcache_flux import apply_dpcache_to_flux
from dpcache.cali_utils import merge_calibration_results
import argparse

def _load_dataset_prompts(dataset_name, sample_rule="random", sample_size=5):
    if dataset_name == "drawbench":
        df = pd.read_csv("datasets/DrawBench.csv")
        prompt_col = "Prompts"
    elif dataset_name == "parti":
        df = pd.read_csv("datasets/PartiPrompts.tsv", sep="\t")
        prompt_col = "Prompt"
    else:
        raise ValueError(f"Dataset not supported: {dataset_name}")

    if isinstance(sample_size, int) and sample_size > 0:
        if sample_rule == "fix":
            prompts = df[prompt_col].tolist()
            prompts = prompts[:: len(prompts) // sample_size][:sample_size]
        else:
            prompts = df[prompt_col].sample(n=sample_size, random_state=42).tolist()
    else:
        prompts = df[prompt_col].tolist()[:50]
    return prompts

def run_flux(
    enable_cache=True,
    model_path="black-forest-labs/FLUX.1-dev",
    num_steps=50,
    order=2,
    cali_prefix="flux_test",
    cali=False,
    k=13,
    first_full_steps=3,
    cost_matrix_path=None,
    dataset=None,
    sample_rule="random",
    sample_size=None,
    output_path=None,
    cost_metric="l1",
):
    operation = "Calibration" if cali else "Inference"
    cache_status = "with DPCache" if enable_cache else "without cache"
    print(f"Starting FLUX {operation} {cache_status}...")
    if enable_cache:
        print(f"Config: steps={num_steps}, order={order}")

    if not dataset:
        input_prompts = [
            "A simple red apple on white background",
        ]
    else:
        input_prompts = _load_dataset_prompts(dataset, sample_rule=sample_rule, sample_size=sample_size)

    print(f"Using {len(input_prompts)} {operation} prompts")

    print("Loading FLUX pipeline...")
    pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)

    pkl_path = f"cost_matrix_{cali_prefix}_{order}.pkl"
    if os.path.exists(pkl_path):
        print(f"Cost matrix file {pkl_path} already exists, removing...")
        os.remove(pkl_path)
    
    if enable_cache:
        pipeline = apply_dpcache_to_flux(
            pipeline,
            mode="Taylor-DP",
            cali=cali,
            num_steps=num_steps,
            order=order,
            k=k,
            first_full_steps=first_full_steps,
            cali_prefix=cali_prefix,
            cost_matrix_path=cost_matrix_path,
            cost_metric=cost_metric,
        )

    pipeline.to("cuda")
    print("Pipeline ready!")
    if not output_path:
        output_path = cali_prefix
    os.makedirs(output_path, exist_ok=True)

    print(f"Running {operation}...")
    for i, prompt in enumerate(tqdm(input_prompts, desc=operation)):
        image = pipeline(prompt, num_inference_steps=num_steps, generator=torch.Generator("cpu").manual_seed(42 + i)).images[0]
        image.save(f"{output_path}/output_{i:02d}.png")

    print(f"{operation} Complete!")

def main():
    parser = argparse.ArgumentParser(description="FLUX DPCache Inference and Calibration")
    parser.add_argument("--mode", choices=["calibrate", "infer"], default="infer",
                        help="Operation mode: calibrate or infer")
    parser.add_argument("--no_cache", action="store_true", default=False,
                        help="Disable cache (baseline mode)")
    parser.add_argument("--cali_prefix", default="flux_test",
                        help="Prefix for calibration files, not used in infer mode")
    parser.add_argument("--k", type=int, default=13,
                        help="Number of steps in sampling schedule, not used in calibrate mode")
    parser.add_argument("--first_full_steps", type=int, default=3,
                        help="Number of initial full steps, not used in calibrate mode")
    parser.add_argument("--cost_matrix_path", default="final_3d_cost_matrix_flux.pkl",
                        help="Path to the cost matrix file for inference, not used in calibrate mode")
    parser.add_argument("--dataset", default="drawbench", choices=["drawbench", "parti"],
                        help="Dataset to use for prompts")
    parser.add_argument("--sample_rule", default="fix", choices=["random", "fix"],
                        help="Sampling rule for dataset")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of samples from dataset")
    parser.add_argument("--output_path",
                        help="Output path for generated images")
    parser.add_argument("--cost_metric", default="l1", choices=["l1", "l2", "cos"],
                        help="Cost metric for calibration: l1 (MAE), l2 (MSE), cos (1-cosine_similarity), not used in infer mode")
    
    args = parser.parse_args()

    enable_cache = not args.no_cache if args.mode == "infer" else True
    
    if args.mode == "calibrate":
        print("Running calibration mode...")
        run_flux(
            enable_cache=True,
            cali_prefix=args.cali_prefix,
            cali=True,
            dataset=args.dataset,
            sample_rule=args.sample_rule,
            sample_size=args.sample_size,
            output_path=args.output_path,
            cost_metric=args.cost_metric,
        )
        
        # Automatically merge calibration results after calibration
        merge_calibration_results(args.cali_prefix)
    elif args.mode == "infer":
        print("Running inference mode...")
        run_flux(
            enable_cache=enable_cache,
            cali=False,
            k=args.k,
            first_full_steps=args.first_full_steps,
            cost_matrix_path=args.cost_matrix_path,
            dataset=args.dataset,
            sample_rule=args.sample_rule,
            sample_size=args.sample_size,
            output_path=args.output_path,
        )

if __name__ == "__main__":
    main()