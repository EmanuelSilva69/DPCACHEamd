import os
import torch
import lpips
import clip
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "resultados_comparacao", "flux")
METODOS = ["baseline", "dpcache", "taylorseer", "teacache"]

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn_alex = lpips.LPIPS(net="alex").to(device)
clip_model, _ = clip.load("ViT-B/32", device=device)


def pegar_imagem(pasta):
    from glob import glob
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        arquivos = glob(os.path.join(pasta, ext))
        if arquivos:
            return arquivos[0]
    return None


def carregar(caminho):
    img = Image.open(caminho).convert("RGB").resize((512, 512))
    return np.array(img), lpips.im2tensor(np.array(img)).to(device)


def clip_score(pil_img, prompt):
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    img = preprocess(pil_img).unsqueeze(0).to(device)
    text = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        img_feat = clip_model.encode_image(img)
        txt_feat = clip_model.encode_text(text)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
    return (img_feat @ txt_feat.T).item()


prompt_file = os.path.join(os.path.dirname(__file__), "prompt_unico.txt")
with open(prompt_file, "r", encoding="utf-8") as f:
    prompt = f.readline().strip()

ref_path = pegar_imagem(os.path.join(BASE, "baseline"))
if ref_path is None:
    print("ERRO: Nenhuma imagem encontrada em baseline/")
    exit(1)

ref_np, ref_t = carregar(ref_path)
ref_pil = Image.open(ref_path).convert("RGB")

print(f"{'Metodo':<20} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'CLIP':>8}")
print("-" * 56)

for metodo in METODOS[1:]:
    img_path = pegar_imagem(os.path.join(BASE, metodo))
    if img_path is None:
        print(f"{metodo:<20} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
        continue

    img_np, img_t = carregar(img_path)
    p = psnr(ref_np, img_np)
    s = ssim(ref_np, img_np, channel_axis=2)
    l = loss_fn_alex(ref_t, img_t).item()
    c = clip_score(Image.open(img_path).convert("RGB"), prompt)

    print(f"{metodo:<20} {p:>8.2f} {s:>8.4f} {l:>8.4f} {c:>8.4f}")
