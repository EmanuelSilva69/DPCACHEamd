## TaylorSeer-Wan2.1

### Multi-GPU inference

#### Sample with Wan-14B

```bash
torchrun --nproc_per_node=8 taylorseer_generate.py \
--task t2v-14B \
--size 1280*720 \
--ckpt_dir ./Wan2.1-T2V-14B \
--dit_fsdp --t5_fsdp \
--ulysses_size 8 \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

#### Sample with Wan-1.3B

```bash
torchrun --nproc_per_node=8 taylorseer_generate.py \
--task t2v-1.3B \
--size 832*480 \
--ckpt_dir ./Wan2.1-T2V-1.3B \
--dit_fsdp --t5_fsdp \
--ulysses_size 8 \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

### Single-GPU inference

```bash
python taylorseer_generate.py \
--task t2v-1.3B \
--size 832*480 \
--ckpt_dir ./Wan2.1-T2V-1.3B \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

It is worth noting that the current TaylorSeer model cannot perform single-GPU inference for 14B models on an A100 with 80GB of memory (multi-GPU inference is supported). If you have such requirements, you may need to consider GPUs with larger memory, such as the H20.


### Change Acceleration ratio

The hyperparameter for the acceleration method in the current version is located at `TaylorSeer/TaylorSeer-Wan2.1/wan/taylorseer/cache_functions/cache_init.py`, Line 49. A larger value of `fresh_threshold` results in a more aggressive acceleration strategy. If a more conservative acceleration is desired to better preserve quality, `fresh_threshold` should be set to a smaller value (e.g., 2 or 3).

