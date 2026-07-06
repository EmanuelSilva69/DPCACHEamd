import os

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_MODELS_DIR = os.getenv('PRETRAINED_MODELS_DIR', os.path.join(BASE_DIR, 'pretrained_models'))

# 模型路径配置
MODEL_PATHS = {
    'text_encoder': os.path.join(PRETRAINED_MODELS_DIR, 'HunyuanVideo/text_encoder'),
    'text_encoder_2': os.path.join(PRETRAINED_MODELS_DIR, 'HunyuanVideo/text_encoder_2'),
    'tokenizer': os.path.join(PRETRAINED_MODELS_DIR, 'HunyuanVideo/tokenizer'),
    'tokenizer_2': os.path.join(PRETRAINED_MODELS_DIR, 'HunyuanVideo/tokenizer_2'),
    'vae': os.path.join(PRETRAINED_MODELS_DIR, 'HunyuanVideo/vae'),
    'feature_extractor': os.path.join(PRETRAINED_MODELS_DIR, 'lllyasviel/flux_redux_bfl'),
    'image_encoder': os.path.join(PRETRAINED_MODELS_DIR, 'lllyasviel/flux_redux_bfl'),
    'transformer': os.path.join(PRETRAINED_MODELS_DIR, 'FramePackI2V_HY'),
}

# 输出目录配置
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# 确保目录存在
def ensure_directories():
    """确保必要的目录存在"""
    os.makedirs(PRETRAINED_MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # 确保模型子目录存在
    for model_path in MODEL_PATHS.values():
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

# 验证模型文件是否存在
def validate_model_paths():
    """验证模型路径是否存在"""
    missing_models = []
    for model_name, model_path in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            missing_models.append(f"{model_name}: {model_path}")
    
    if missing_models:
        print("警告：以下模型路径不存在：")
        for missing in missing_models:
            print(f"  - {missing}")
        print("\n请确保已下载所有必要的模型文件。")
        return False
    return True 