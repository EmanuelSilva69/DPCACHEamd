## TaylorSeer-DiT

### 1. Prepare Environment

```bash
cd TaylorSeer-DiT
conda env create -f environment.yml
conda activate DiT
pip install flash-attention
```

### 2. Download Checkpoints

Simply follow the official documentation to download the necessary checkpoints.

### 3. Run Samples

#### Single-Batch Inference

Set your desired class ID in `TaylorSeer-DiT/sample.py`.

<details>
  <summary>Recommended Parameter Settings</summary>
  
  It is recommended to use `interval=4` and `max_order=4`, as this configuration achieves **3.57× computation compression** while only increasing FID by **0.17**. In contrast, previous methods at the same compression level resulted in an **FID increase of at least 1.10**.  
  You can also experiment with different parameter configurations to suit your needs. For example:  
  - `interval=3, max_order=3`: FID is **2.34**, compared to **2.32** without acceleration, showing almost no change.
</details>

Run the inference with the recommended settings:

```bash
python sample.py --ddim-sample --num-sampling-steps 50 --interval 4 --max-order 4 
```

#### Distributed Data Parallel (DDP) Inference

```bash
torchrun --nnodes=1 --nproc_per_node=8 sample_ddp.py \
  --model DiT-XL/2 \
  --per-proc-batch-size 50 \
  --image-size 256 \
  --cfg-scale 1.5 \
  --ddim-sample \
  --num-sampling-steps 50 \
  --interval 4 \
  --max-order 4 \
  --num-fid-samples 50000
```