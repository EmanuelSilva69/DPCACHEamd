import torch
import os
from pathlib import Path
from tqdm import tqdm
from wan_pipeline import WanI2VPipeline
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel, AutoTokenizer, UMT5EncoderModel, CLIPImageProcessor
from diffusers.models import WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from dpcache_wan import apply_dpcache_to_wan
from dpcache.cali_utils import merge_calibration_results
import argparse


def _load_image_dataset(dataset_dir, sample_rule="random", sample_size=10):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(dataset_path.glob(ext)))

    if len(image_files) == 0:
        raise ValueError(f"No image files found in {dataset_dir}")

    if isinstance(sample_size, int) and sample_size > 0:
        if sample_rule == "fix":
            step = len(image_files) // sample_size
            sampled_files = image_files[::step][:sample_size]
        else:
            import random
            random.seed(42)
            sampled_files = random.sample(image_files, min(sample_size, len(image_files)))
    else:
        sampled_files = image_files[:50]

    samples = [(str(f), f.stem) for f in sampled_files]

    return samples


def run_wan_i2v(
    enable_cache=True,
    model_id="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    image_path="test.jpg",
    prompt="a blue car driving down a dirt road near train tracks",
    num_steps=50,
    order=2,
    cali_prefix="wan_test",
    cali=False,
    k=11,
    first_full_steps=4,
    last_full_steps=1,
    cost_matrix_path="final_3d_cost_matrix_wan_cfg_3.pkl",
    dataset_dir=None,
    sample_rule="random",
    sample_size=None,
    output_path=None,
    height=832,
    width=832,
    num_frames=81,
    guidance_scale=3.0,
    fps=16,
):
    operation = "Calibration" if cali else "Inference"
    cache_status = "with DPCache" if enable_cache else "without cache"
    print(f"Starting Wan I2V {operation} {cache_status}...")
    if enable_cache:
        print(f"Config: steps={num_steps}, order={order}, height={height}, width={width}, frames={num_frames}, cfg={guidance_scale}")

    if not dataset_dir:
        input_samples = [(image_path, prompt)]
    else:
        input_samples = _load_image_dataset(dataset_dir, sample_rule=sample_rule, sample_size=sample_size)
    
    print(f"Using {len(input_samples)} {operation} samples")

    print("Loading Wan pipeline...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=torch.float32
    )
    image_processor = CLIPImageProcessor.from_pretrained(
        model_id, subfolder="image_processor"
    )
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    )
    transformer = WanTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler"
    )
    
    # Remove existing cost matrix file if calibrating
    if cali:
        pkl_path = f"cost_matrix_{cali_prefix}_{order}.pkl"
        if os.path.exists(pkl_path):
            print(f"Cost matrix file {pkl_path} already exists, removing...")
            os.remove(pkl_path)
    
    pipeline = WanI2VPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        image_processor=image_processor,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
    )

    if enable_cache:
        pipeline = apply_dpcache_to_wan(
            pipeline,
            mode="Taylor-DP",
            cali=cali,
            num_steps=num_steps,
            order=order,
            k=k,
            first_full_steps=first_full_steps,
            last_full_steps=last_full_steps,
            cali_prefix=cali_prefix,
            cost_matrix_path=cost_matrix_path,
        )

    pipeline.to("cuda")
    print("Pipeline ready!")

    if not output_path:
        output_path = cali_prefix if cali else "wan_output"
    os.makedirs(output_path, exist_ok=True)
    
    negative_prompt = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
        "paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, "
        "ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
        "misshapen limbs, fused fingers, still picture, messy background, three legs, "
        "many people in the background, walking backwards"
    )

    print(f"Running {operation}...")
    for i, (img_path, sample_prompt) in enumerate(tqdm(input_samples, desc=operation)):
        input_image = load_image(img_path).resize((width, height))
        
        video = pipeline(
            image=input_image,
            prompt=sample_prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator("cuda").manual_seed(42 + i),
        ).frames[0]

        export_to_video(video, f"{output_path}/output_{i:02d}.mp4", fps=fps)

    print(f"{operation} Complete!")


def main():
    parser = argparse.ArgumentParser(description="Wan I2V DPCache Inference and Calibration")
    parser.add_argument("--mode", choices=["calibrate", "infer"], default="infer",
                        help="Operation mode: calibrate or infer")
    parser.add_argument("--no_cache", action="store_true", default=False,
                        help="Disable cache (baseline mode)")
    parser.add_argument("--cali_prefix", default="wan_test",
                        help="Prefix for calibration files, not used in infer mode")
    parser.add_argument("--dataset_dir", default=None,
                        help="Directory containing images (filenames are prompts)")
    parser.add_argument("--sample_rule", default="fix", choices=["random", "fix"],
                        help="Sampling rule for dataset")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of samples from dataset")
    parser.add_argument("--image_path", default="test.jpg",
                        help="Path to single input image (used when dataset_dir is not specified)")
    parser.add_argument("--prompt", default="a blue car driving down a dirt road near train tracks",
                        help="Text prompt (used when dataset_dir is not specified)")
    parser.add_argument("--k", type=int, default=12,
                        help="Number of steps in sampling schedule, not used in calibrate mode")
    parser.add_argument("--first_full_steps", type=int, default=4,
                        help="Number of initial full steps, not used in calibrate mode")
    parser.add_argument("--last_full_steps", type=int, default=1,
                        help="Number of final full steps, not used in calibrate mode")
    parser.add_argument("--cost_matrix_path", default="final_3d_cost_matrix_wan_cfg_3.pkl",
                        help="Path to the cost matrix file, not used in calibrate mode")
    parser.add_argument("--output_path",
                        help="Output path for generated videos")
    parser.add_argument("--height", type=int, default=832,
                        help="Video height")
    parser.add_argument("--width", type=int, default=832,
                        help="Video width")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of video frames")
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                        help="CFG scale")
    parser.add_argument("--fps", type=int, default=16,
                        help="Output video FPS")

    args = parser.parse_args()

    enable_cache = not args.no_cache if args.mode == "infer" else True

    if args.mode == "calibrate":
        print("Running calibration mode...")
        run_wan_i2v(
            enable_cache=True,
            cali=True,
            cali_prefix=args.cali_prefix,
            dataset_dir=args.dataset_dir,
            sample_rule=args.sample_rule,
            sample_size=args.sample_size,
            output_path=args.output_path,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            fps=args.fps,
        )
        
        merge_calibration_results(args.cali_prefix)
    elif args.mode == "infer":
        print("Running inference mode...")
        run_wan_i2v(
            enable_cache=enable_cache,
            cali=False,
            k=args.k,
            first_full_steps=args.first_full_steps,
            last_full_steps=args.last_full_steps,
            cost_matrix_path=args.cost_matrix_path,
            dataset_dir=args.dataset_dir,
            sample_rule=args.sample_rule,
            sample_size=args.sample_size,
            image_path=args.image_path,
            prompt=args.prompt,
            output_path=args.output_path,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
