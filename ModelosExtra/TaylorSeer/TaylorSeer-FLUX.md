## TaylorSeer-FLUX

### 1. Set Up Conda Environment

Follow the official documentation instructions to create the Conda environment:

```bash
conda create -n flux python=3.10
conda activate flux
pip install -e ".[all]"
```

### 2. Download Checkpoints with Your Hugging Face Token

If you experience connection issues with Hugging Face, you can use the Hugging Face mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Make sure you have obtained the necessary permissions and exported your token:

```bash
huggingface-cli download --token YOUR_HF_TOKEN --resume-download black-forest-labs/FLUX.1-dev --local-dir /path/to/save/pretrained_models/black-forest-labs/FLUX.1-dev
huggingface-cli download --token YOUR_HF_TOKEN --resume-download black-forest-labs/FLUX.1-schnell --local-dir /path/to/save/pretrained_models/black-forest-labs/FLUX.1-schnell
huggingface-cli download --token YOUR_HF_TOKEN --resume-download google/t5-v1_1-xxl --local-dir /path/to/save/pretrained_models/google/t5-v1_1-xxl
huggingface-cli download --token YOUR_HF_TOKEN --resume-download openai/clip-vit-large-patch14 --local-dir /path/to/save/pretrained_models/openai/clip-vit-large-patch14
```

<details>
  <summary>Download Checkpoints on AutoDL</summary>
  
  ```bash
  huggingface-cli download --token YOUR_HF_TOKEN --resume-download black-forest-labs/FLUX.1-dev --local-dir /root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-dev
  huggingface-cli download --token YOUR_HF_TOKEN --resume-download black-forest-labs/FLUX.1-schnell --local-dir /root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-schnell
  huggingface-cli download --token YOUR_HF_TOKEN --resume-download google/t5-v1_1-xxl --local-dir /root/autodl-tmp/pretrained_models/google/t5-v1_1-xxl
  huggingface-cli download --token YOUR_HF_TOKEN --resume-download openai/clip-vit-large-patch14 --local-dir /root/autodl-tmp/pretrained_models/openai/clip-vit-large-patch14
  ```
</details>

### 3. Set Environment Variables (in `.bashrc` file)

```bash
export FLUX_SCHNELL="/path/to/save/pretrained_models/black-forest-labs/FLUX.1-schnell/flux1-schnell.safetensors"
export FLUX_DEV="/path/to/save/pretrained_models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors"
export AE="/path/to/save/pretrained_models/black-forest-labs/FLUX.1-dev/ae.safetensors"
```

<details>
  <summary>Set Environment Variables for AutoDL</summary>
  
  ```bash
  export FLUX_SCHNELL="/root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-schnell/flux1-schnell.safetensors"
  export FLUX_DEV="/root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors"
  export AE="/root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-dev/ae.safetensors"
  ```
</details>

### 4. Sampling with TaylorSeer-FLUX

#### Interactive Sampling (Should Download Extra NSFW Classifier)

```bash
python -m flux --name <name> --loop
```

#### Single Sample Generation (Should Download Extra NSFW Classifier)

```bash
python -m flux --name <name> \
  --height <height> --width <width> \
  --prompt "<prompt>"
```

Typically, `<name>` should be set to `flux-dev`.

#### Batch Sampling with Prompt File (Recommended)

```bash
python src/sample.py --prompt_file </path/to/your/prompt.txt> \
  --width 1024 --height 1024 --model_name flux-dev \
  --add_sampling_metadata --output_dir </path/to/your/generated/samples/folder> --num_steps 50
```

The `--add_sampling_metadata` parameter determines whether the prompt is embedded in the image's EXIF metadata.

#### FLOPs Testing (No Image Generation)

```bash
python src/sample.py --prompt_file </path/to/your/test/prompt.txt> \
  --width 1024 --height 1024 --model_name flux-dev \
  --add_sampling_metadata --output_dir </path/to/your/generated/samples/folder> \
  --num_steps 50 --test_FLOPs
```

> **Note:** When testing FLOPs, ensure that `TaylorSeer-FLUX/src/flux/math.py` uses a naive computation method with FlashAttention disabled. Specifically, disable `torch.nn.functional.scaled_dot_product_attention` and switch to the `dot_product_attention` function provided in the file.

### 5. Modify Configuration for Custom Needs

The current framework supports testing multiple methods. You can add your own methods or modify existing ones in:

`TaylorSeer-FLUX/src/flux/modules/cache_functions/cache_init.py`

For example, the TaylorSeer method allows control over the acceleration ratio by adjusting the `fresh_threshold` parameter at **line 94**. Additionally, you can modify the `max_order` parameter at **line 98** to use different orders of TaylorSeer.

The default parameters are already optimized to balance quality and acceleration, but you are free to experiment with different configurations.

