# TaylorSeer-FramePack

## 1. Download Required Models

You need to download the following models using `huggingface-cli`:

```bash
# Create model directory
mkdir -p pretrained_models

# Download HunyuanVideo models
huggingface-cli download --resume-download hunyuanvideo-community/HunyuanVideo --local-dir pretrained_models/HunyuanVideo
 
# Download FLUX models
huggingface-cli download --resume-download lllyasviel/flux_redux_bfl --local-dir pretrained_models/lllyasviel/flux_redux_bfl

# Download FramePack main model
huggingface-cli download --resume-download lllyasviel/FramePackI2V_HY --local-dir pretrained_models/FramePackI2V_HY
```

If you have slow download speeds, you can use a mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 2. (Optional) Install Attention Acceleration Libraries

For best performance, it is recommended to install one or more of the following libraries:

```bash
# xformers (recommended for most users)
pip install xformers

# flash-attn (for supported CUDA versions)
pip install flash-attn

# sageattention (Linux only)
pip install sageattention==1.0.6
```

You can install any combination of these libraries, or skip this step if you do not need advanced attention acceleration.

## 3. Launch the Application

Install dependencies first:

```bash
pip install -r requirements.txt
```

Then start the Gradio demo:

```bash
python demo_gradio.py
```

You can specify options such as port or browser auto-launch:

```bash
python demo_gradio.py --port 7860 --inbrowser
```

## 4. Accelerate with TaylorSeer

TaylorSeer-FramePack integrates Taylor series-based acceleration for next-frame video generation. The acceleration is enabled by default and requires no extra configuration. The system will automatically:

- Use Taylor series prediction for efficient inference
- Dynamically manage memory and cache for optimal speed
- Adjust acceleration strategies based on your hardware

**No manual steps are needed to enable TaylorSeer acceleration.**

For best performance, ensure your GPU drivers and CUDA are up to date. If you encounter memory issues, try reducing the `latent_window_size` parameter or use a smaller model.

## 5. Configure TaylorSeer Parameters

You can customize TaylorSeer acceleration by modifying the cache parameters in `diffusers_helper/models/cache_functions/cache_init.py`:

```python
cache_dic['fresh_threshold'] = 4  # Cache refresh interval
cache_dic['max_order'] = 1        # Maximum Taylor series order
```

