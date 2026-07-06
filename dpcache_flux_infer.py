import os
import sys
import argparse
import torch

torch.manual_seed(42)

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

from diffusers import DiffusionPipeline
from dpcache_flux import apply_dpcache_to_flux


MODEL_PATH_DEFAULT = (
    "C:\\Users\\Emanuel\\.cache\\huggingface\\hub\\"
    "models--Freepik--flux.1-lite-8B-alpha\\snapshots\\"
    "812d376439b6e37b0e6f6dd401b2a98b1effacdb"
)


def load_prompts(prompt_file=None, prompt=None):
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if prompt:
        return [prompt]
    return ["A simple red apple on white background"]


def run_flux(
    enable_cache=True,
    model_path=MODEL_PATH_DEFAULT,
    num_steps=50,
    order=2,
    cali_prefix="flux_test",
    cali=False,
    k=13,
    first_full_steps=3,
    cost_matrix_path=None,
    prompts=None,
    output_path=None,
    cost_metric="l1",
    seed=42,
):
    operation = "Calibration" if cali else "Inference"
    cache_status = "with DPCache" if enable_cache else "without cache"
    print(f"Starting FLUX {operation} {cache_status}...")
    if enable_cache:
        print(f"Config: steps={num_steps}, order={order}")

    print(f"Using {len(prompts)} {operation} prompt(s)")
    print("Loading FLUX pipeline...")

    if not os.path.exists(model_path):
        print(
            f"AVISO: Caminho do modelo nao encontrado: {model_path}"
        )
        print("Verifique se o snapshot hash esta correto em MODEL_PATH_DEFAULT.")
        sys.exit(1)

    try:
        pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    except OSError as e:
        print(f"ERRO: Nao foi possivel carregar o modelo local em:\n  {model_path}")
        print(f"Detalhes: {e}")
        print("Desative 'local_files_only=True' ou baixe o modelo manualmente.")
        sys.exit(1)

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

    pipeline.enable_model_cpu_offload()
    pipeline.enable_vae_slicing()
    pipeline.enable_vae_tiling()
    print("Pipeline ready!")

    if not output_path:
        output_path = cali_prefix
    os.makedirs(output_path, exist_ok=True)

    for i, prompt in enumerate(prompts):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            image = pipeline(
                prompt,
                num_inference_steps=num_steps,
                generator=torch.Generator("cpu").manual_seed(seed + i),
            ).images[0]
        image.save(f"{output_path}/output_{i:02d}.png")
        torch.cuda.empty_cache()

    print(f"{operation} Complete!")


def main():
    parser = argparse.ArgumentParser(
        description="FLUX DPCache Inference with local model"
    )
    parser.add_argument(
        "--model_path",
        default=MODEL_PATH_DEFAULT,
        help="Caminho local para o snapshot do modelo FLUX",
    )
    parser.add_argument(
        "--prompt_file",
        default=None,
        help="Arquivo com prompts (um por linha)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt unico para geracao",
    )
    parser.add_argument(
        "--mode",
        choices=["calibrate", "infer"],
        default="infer",
        help="Operation mode: calibrate or infer",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        default=False,
        help="Disable cache (baseline mode)",
    )
    parser.add_argument(
        "--cali_prefix",
        default="flux_test",
        help="Prefix for calibration files",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=13,
        help="Number of steps in sampling schedule",
    )
    parser.add_argument(
        "--first_full_steps",
        type=int,
        default=3,
        help="Number of initial full steps",
    )
    parser.add_argument(
        "--cost_matrix_path",
        default="final_3d_cost_matrix_flux.pkl",
        help="Path to the cost matrix file for inference",
    )
    parser.add_argument(
        "--output_path", help="Output path for generated images"
    )
    parser.add_argument(
        "--cost_metric",
        default="l1",
        choices=["l1", "l2", "cos"],
        help="Cost metric for calibration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )

    args = parser.parse_args()

    prompts = load_prompts(args.prompt_file, args.prompt)

    enable_cache = not args.no_cache if args.mode == "infer" else True

    if args.mode == "calibrate":
        print("Running calibration mode...")
        run_flux(
            enable_cache=True,
            model_path=args.model_path,
            cali_prefix=args.cali_prefix,
            cali=True,
            prompts=prompts,
            output_path=args.output_path,
            cost_metric=args.cost_metric,
            seed=args.seed,
        )
        from dpcache.cali_utils import merge_calibration_results
        merge_calibration_results(args.cali_prefix)
    elif args.mode == "infer":
        print("Running inference mode...")
        run_flux(
            enable_cache=enable_cache,
            model_path=args.model_path,
            cali=False,
            num_steps=args.num_steps,
            k=args.k,
            first_full_steps=args.first_full_steps,
            cost_matrix_path=args.cost_matrix_path,
            prompts=prompts,
            output_path=args.output_path,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
