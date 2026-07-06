import os
import time
import argparse
from dataclasses import dataclass
from tqdm import tqdm

import torch
from PIL import Image, ExifTags
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

from hi_diffusers import HiDreamImagePipeline
from hi_diffusers import HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler


@dataclass
class SamplingOptions:
    prompts: list[str]          # List of prompts
    width: int                  # Image width
    height: int                 # Image height
    num_steps: int              # Number of sampling steps
    guidance: float             # Guidance scale
    seed: int | None            # Random seed
    num_images_per_prompt: int  # Number of images per prompt
    batch_size: int             # Batch size (for prompts)
    model_type: str             # Model type
    output_dir: str             # Output directory
    add_sampling_metadata: bool # Whether to add metadata
    use_nsfw_filter: bool       # Whether to enable NSFW filter


# Model configuration
MODEL_CONFIGS = {
    "dev": {
        #"path": '/root/autodl-tmp/pretrained_models/HiDream-ai/HiDream-I1-Dev', # Hard Coded path for local loading
        "path": 'HiDream-ai/HiDream-I1-Dev',
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    },
    "full": {
        #"path": '/root/autodl-tmp/pretrained_models/HiDream-ai/HiDream-I1-Full', # Hard Coded path for local loading
        "path": 'HiDream-ai/HiDream-I1-Full',
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler
    },
    "fast": {
        "path": 'HiDream-ai/HiDream-I1-Fast',
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    }
}


#LLAMA_MODEL_NAME = "/root/autodl-tmp/pretrained_models/Meta-Llama-3.1-8B" # Hard Coded path for local loading
LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B"  # Hugging Face Hub loading
NSFW_THRESHOLD = 0.85  # NSFW score threshold


def load_models(model_type):
    """Load models."""
    if model_type not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model type: {model_type}, available options: {available}")
    
    config = MODEL_CONFIGS[model_type]
    pretrained_model_name_or_path = config["path"]
    scheduler = config["scheduler"](
        num_train_timesteps=1000, 
        shift=config["shift"], 
        use_dynamic_shifting=False
    )
    
    print(f"Loading {model_type} model...")
    
    # Load tokenizer and text encoder
    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
        LLAMA_MODEL_NAME,
        use_fast=False
    )
    
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        output_hidden_states=True,
        output_attentions=True,
        return_dict_in_generate=True, 
        torch_dtype=torch.bfloat16
    ).to("cuda")

    # Load the transformer and pipeline
    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="transformer", 
        torch_dtype=torch.bfloat16
    ).to("cuda")

    pipe = HiDreamImagePipeline.from_pretrained(
        pretrained_model_name_or_path, 
        scheduler=scheduler,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16
    ).to("cuda", torch.bfloat16)
    pipe.transformer = transformer
    
    return pipe


def read_prompts(prompt_file: str):
    """Read prompts from a file."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def main(opts: SamplingOptions):
    """Main function handling image generation logic."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Optional NSFW classifier
    if opts.use_nsfw_filter:
        try:
            from transformers import pipeline
            nsfw_classifier = pipeline(
                "image-classification",
                model="Falconsai/nsfw_image_detection",
                device=device
            )
        except Exception as e:
            print(f"Warning: Unable to load NSFW filter: {e}")
            nsfw_classifier = None
    else:
        nsfw_classifier = None

    # Load model
    model_type = opts.model_type
    pipe = load_models(model_type)
    
    # Ensure width and height are multiples of 16
    opts.width = 16 * (opts.width // 16)
    opts.height = 16 * (opts.height // 16)

    # Set up output directory and index
    output_name = os.path.join(opts.output_dir, f"img_{{idx}}.jpg")
    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)
    idx = 0  # Image index

    # Set random seed
    if opts.seed is not None:
        base_seed = opts.seed
    else:
        base_seed = torch.randint(0, 2**32, (1,)).item()

    prompts = opts.prompts
    total_images = len(prompts) * opts.num_images_per_prompt

    progress_bar = tqdm(total=total_images, desc="Generating images") 

    # Calculate number of prompt batches
    num_prompt_batches = (len(prompts) + opts.batch_size - 1) // opts.batch_size

    for batch_idx in range(num_prompt_batches):
        prompt_start = batch_idx * opts.batch_size
        prompt_end = min(prompt_start + opts.batch_size, len(prompts))
        batch_prompts = prompts[prompt_start:prompt_end]
        num_prompts_in_batch = len(batch_prompts)

        # Generate images for each prompt
        for image_idx in range(opts.num_images_per_prompt):
            for i, prompt in enumerate(batch_prompts):
                current_idx = idx + i
                seed = base_seed + current_idx  # Assign a unique seed per image
                generator = torch.Generator("cuda").manual_seed(seed)
                
                # Generate image
                with torch.no_grad():
                    image = pipe(
                        prompt,
                        height=opts.height,
                        width=opts.width,
                        guidance_scale=opts.guidance,
                        num_inference_steps=opts.num_steps,
                        num_images_per_prompt=1,
                        generator=generator
                    ).images[0]

                # Optional NSFW filtering
                nsfw_score = 0.0
                if nsfw_classifier is not None:
                    try:
                        nsfw_result = nsfw_classifier(image)
                        nsfw_score = next((res["score"] for res in nsfw_result if res["label"] == "nsfw"), 0.0)
                    except Exception as e:
                        print(f"NSFW classification error: {e}")

                if nsfw_score < NSFW_THRESHOLD:
                    # Add EXIF data
                    exif_data = Image.Exif()
                    exif_data[ExifTags.Base.Software] = "AI generated;txt2img;hidream"
                    exif_data[ExifTags.Base.Make] = "HiDream-ai"
                    exif_data[ExifTags.Base.Model] = opts.model_type
                    if opts.add_sampling_metadata:
                        exif_data[ExifTags.Base.ImageDescription] = prompt
                    
                    # Save image
                    fn = output_name.format(idx=current_idx)
                    image.save(fn, exif=exif_data, quality=95, subsampling=0)
                else:
                    print("Generated image may contain inappropriate content; skipped.")

                progress_bar.update(1)
            
            idx += num_prompts_in_batch  # Update image index

    progress_bar.close()


def app():
    """Command-line application entry point."""
    parser = argparse.ArgumentParser(description="Generate images using the HiDream model.")
    parser.add_argument('--prompt_file', type=str, default='/path/to/your/prompt.txt', help='Path to the prompt text file.')
    parser.add_argument('--width', type=int, default=1024, help='Width of the generated images.')
    parser.add_argument('--height', type=int, default=1024, help='Height of the generated images.')
    parser.add_argument('--num_steps', type=int, default=None, help='Number of sampling steps.')
    parser.add_argument('--guidance', type=float, default=None, help='Guidance scale.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--num_images_per_prompt', type=int, default=1, help='Number of images per prompt.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for prompts.')
    parser.add_argument('--model_type', type=str, default='full', choices=['dev', 'full', 'fast'], help='Type of model to use.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save generated images.')
    parser.add_argument('--add_sampling_metadata', action='store_true', help='Whether to add prompt metadata to images.')
    parser.add_argument('--use_nsfw_filter', action='store_true', help='Enable NSFW filtering.')

    args = parser.parse_args()

    # Read prompts
    prompts = read_prompts(args.prompt_file)
    
    # Use default model settings if not specified
    model_config = MODEL_CONFIGS[args.model_type]
    if args.num_steps is None:
        args.num_steps = model_config["num_inference_steps"]
    if args.guidance is None:
        args.guidance = model_config["guidance_scale"]

    # Create sampling options
    opts = SamplingOptions(
        prompts=prompts,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        guidance=args.guidance,
        seed=args.seed,
        num_images_per_prompt=args.num_images_per_prompt,
        batch_size=args.batch_size,
        model_type=args.model_type,
        output_dir=args.output_dir,
        add_sampling_metadata=args.add_sampling_metadata,
        use_nsfw_filter=args.use_nsfw_filter,
    )

    # Run main function
    main(opts)


if __name__ == '__main__':
    app()
