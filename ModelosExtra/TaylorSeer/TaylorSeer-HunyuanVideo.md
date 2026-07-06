## **TaylorSeer-HunyuanVideo**

### **1. Prepare Environment**

Follow the official **HunyuanVideo** documentation to set up the environment.

<details>
  <summary><strong>Conda Environment Setup</strong></summary>

  ```bash
  # 1. Create the Conda environment
  conda create -n HunyuanVideo python==3.10.9

  # 2. Activate the environment
  conda activate HunyuanVideo

  # 3. Install PyTorch and dependencies
  # For CUDA 11.8
  conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia
  # For CUDA 12.4
  conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

  # 4. Install required Python dependencies
  python -m pip install -r requirements.txt

  # 5. Install FlashAttention v2 for acceleration (requires CUDA 11.8 or later)
  python -m pip install ninja
  python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3

  # 6. Install xDiT for parallel inference (recommended with PyTorch 2.4.0 and FlashAttention 2.6.3)
  python -m pip install xfuser==0.4.0
  ```

  If you encounter a **floating point exception (core dump)** on specific GPUs, try the following solutions:

  ```bash
  # Option 1: Ensure CUDA 12.4, CUBLAS>=12.4.5.8, and CUDNN>=9.00 are installed
  # (Alternatively, use our prebuilt CUDA 12 Docker image)
  pip install nvidia-cublas-cu12==12.4.5.8
  export LD_LIBRARY_PATH=/opt/conda/lib/python3.8/site-packages/nvidia/cublas/lib/

  # Option 2: Force using the CUDA 11.8-compiled version of PyTorch and dependencies
  pip uninstall -r requirements.txt  # Uninstall all packages
  pip uninstall -y xfuser
  pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
  pip install -r requirements.txt
  pip install ninja
  pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3
  pip install xfuser==0.4.0
  ```

</details>

### **2. Download Checkpoints**

Refer to the [**checkpoint download guide**](TaylorSeer-HunyuanVideo/ckpts/README.md)
---

### **3. Run TaylorSeer-HunyuanVideo Samples**

#### **Single Video Inference**

Run inference on a single video. Feel free to adjust the parameters and prompt as needed.

```bash
cd HunyuanVideo
python3 sample_video.py \
  --video-size 480 640 \
  --video-length 65 \
  --infer-steps 50 \
  --seed 42 \
  --prompt "A cat walks on the grass, realistic style." \
  --flow-reverse \
  --use-cpu-offload \
  --save-path /path/to/save/videos
```

---

#### **Multi-Video Inference (VBench Testing)**

We provide a script to test **VBench** on HunyuanVideo, supporting multi-GPU parallel inference.

```bash
cd HunyuanVideo
# Run VBench evaluation:
# ./sample_vbench.sh <full_info_path> <Num_Devices> <SEED> <Num_Samples> <Video_Save_Path> <Path2Log>
./eval/sample_vbench.sh ./eval/ 1 42 5 /path/to/save/vbench/videos /path/to/save/logger/files
```

---

<details>
  <summary><strong>Hyperparameter Tuning & Recommendations</strong></summary>

  TaylorSeer-HunyuanVideo is evaluated with the default parameters:  
  - `--video-size 480 640`
  - `--video-length 65`

  However, you can adjust these configurations based on your requirements.

  **Modifying TaylorSeer Method Parameters**  
  The method-specific configurations can be adjusted in:

  ```
  TaylorSeer-HunyuanVideo/hyvideo/modules/cache_functions/cache_init.py
  ```

  The hyperparameters for different methods are defined between **line 65 and line 114**.  
  To switch between different methods, modify **line 64** by setting `mode` to the corresponding method name.

  > **Example: Default Taylor Mode**  
  If `mode="Taylor"`, then the configuration from **line 103 to line 114** is applied.

  - **Adjusting `fresh_threshold` (Line 109)**:  
    - This parameter **directly impacts acceleration**.  
    - A **higher `fresh_threshold`** results in **faster inference** but may reduce generation quality.  
    - We've tested setting it up to **10**, which significantly speeds up processing but leads to noticeable quality degradation—though objects in the video remain recognizable.

  - **Setting `max_order` (Line 110)**:  
    - This controls the **maximum order of the Taylor series expansion**.  
    - We recommend setting this to **1** for a good balance between speed and quality.

</details>

---

<details>
  <summary><strong>About Generation Quality</strong></summary>

  Compared to other methods, **TaylorSeer preserves more details** in videos, achieving **better quality**.  
  However, like most acceleration techniques, it **does not guarantee identical outputs** compared to the non-accelerated version.  
  You may need to **fine-tune parameters** based on your specific requirements.

</details>
