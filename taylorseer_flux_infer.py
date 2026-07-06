import os
import sys
import math
import argparse
import importlib.util

import torch

torch.manual_seed(42)

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import numpy as np
from diffusers import DiffusionPipeline
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers

logger = logging.get_logger(__name__)

TAYLORSEER_DIR = os.path.join(
    os.path.dirname(__file__),
    "ModelosExtra",
    "TaylorSeer",
    "TaylorSeers-Diffusers",
    "taylorseer_flux",
)
sys.path.insert(0, TAYLORSEER_DIR)

from cache_functions import cache_init, cal_type
from taylorseer_utils import derivative_approximation, taylor_formula, taylor_cache_init

forwards_dir = os.path.join(TAYLORSEER_DIR, "forwards")
_double_spec = importlib.util.spec_from_file_location(
    "ts_double_forward", os.path.join(forwards_dir, "double_transformer_forward.py")
)
_double_mod = importlib.util.module_from_spec(_double_spec)
_double_spec.loader.exec_module(_double_mod)
taylorseer_flux_double_block_forward = _double_mod.taylorseer_flux_double_block_forward

_single_spec = importlib.util.spec_from_file_location(
    "ts_single_forward", os.path.join(forwards_dir, "single_transformer_forward.py")
)
_single_mod = importlib.util.module_from_spec(_single_spec)
_single_spec.loader.exec_module(_single_mod)
taylorseer_flux_single_block_forward = _single_mod.taylorseer_flux_single_block_forward

MODEL_PATH_DEFAULT = (
    "C:\\Users\\Emanuel\\.cache\\huggingface\\hub\\"
    "models--Freepik--flux.1-lite-8B-alpha\\snapshots\\"
    "812d376439b6e37b0e6f6dd401b2a98b1effacdb"
)


def taylorseer_flux_forward(
    self: FluxTransformer2DModel,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: dict | None = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
):
    if joint_attention_kwargs is None:
        joint_attention_kwargs = {}
    if joint_attention_kwargs.get("cache_dic", None) is None:
        joint_attention_kwargs["cache_dic"], joint_attention_kwargs["current"] = (
            cache_init(self)
        )

    cal_type(joint_attention_kwargs["cache_dic"], joint_attention_kwargs["current"])

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    else:
        if (
            joint_attention_kwargs is not None
            and joint_attention_kwargs.get("scale", None) is not None
        ):
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` "
                "when not using the PEFT backend is ineffective."
            )

    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if (
        joint_attention_kwargs is not None
        and "ip_adapter_image_embeds" in joint_attention_kwargs
    ):
        ip_adapter_image_embeds = joint_attention_kwargs.pop(
            "ip_adapter_image_embeds"
        )
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    joint_attention_kwargs["current"]["stream"] = "double_stream"

    for index_block, block in enumerate(self.transformer_blocks):
        joint_attention_kwargs["current"]["layer"] = index_block

        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs: dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            encoder_hidden_states, hidden_states = (
                torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        if controlnet_block_samples is not None:
            interval_control = (
                len(self.transformer_blocks) / len(controlnet_block_samples)
            )
            interval_control = int(np.ceil(interval_control))
            if controlnet_blocks_repeat:
                hidden_states = (
                    hidden_states
                    + controlnet_block_samples[
                        index_block % len(controlnet_block_samples)
                    ]
                )
            else:
                hidden_states = (
                    hidden_states
                    + controlnet_block_samples[
                        index_block // interval_control
                    ]
                )

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    joint_attention_kwargs["current"]["stream"] = "single_stream"

    for index_block, block in enumerate(self.single_transformer_blocks):
        joint_attention_kwargs["current"]["layer"] = index_block

        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs: dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                temb,
                image_rotary_emb,
                **ckpt_kwargs,
            )
        else:
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        if controlnet_single_block_samples is not None:
            interval_control = (
                len(self.single_transformer_blocks)
                / len(controlnet_single_block_samples)
            )
            interval_control = int(np.ceil(interval_control))
            hidden_states[
                :, encoder_hidden_states.shape[1] :, ...
            ] = (
                hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                + controlnet_single_block_samples[
                    index_block // interval_control
                ]
            )

    hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    joint_attention_kwargs["current"]["step"] += 1

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


def load_prompts(prompt_file=None, prompt=None):
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if prompt:
        return [prompt]
    return ["A simple red apple on white background"]


def main():
    parser = argparse.ArgumentParser(
        description="FLUX TaylorSeer Inference with local model (diffusers)"
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
        "--output",
        default="output_taylorseer.png",
        help="Caminho para salvar a imagem gerada",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Numero de passos de inferencia",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()
    prompts = load_prompts(args.prompt_file, args.prompt)

    if not os.path.exists(args.model_path):
        print(
            f"AVISO: Caminho do modelo nao encontrado: {args.model_path}"
        )
        print("Verifique se o snapshot hash esta correto.")
        sys.exit(1)

    try:
        pipeline = DiffusionPipeline.from_pretrained(
            args.model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
    except OSError as e:
        print(
            f"ERRO: Nao foi possivel carregar o modelo local em:\n  {args.model_path}"
        )
        print(f"Detalhes: {e}")
        sys.exit(1)

    pipeline.transformer.__class__.num_steps = args.steps
    pipeline.transformer.__class__.forward = taylorseer_flux_forward

    for double_block in pipeline.transformer.transformer_blocks:
        double_block.__class__.forward = taylorseer_flux_double_block_forward

    for single_block in pipeline.transformer.single_transformer_blocks:
        single_block.__class__.forward = taylorseer_flux_single_block_forward

    pipeline.enable_model_cpu_offload()
    out_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
    os.makedirs(out_dir, exist_ok=True)
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prompt[:60]}...")
        img = pipeline(
            prompt,
            num_inference_steps=args.steps,
            generator=torch.Generator("cpu").manual_seed(args.seed + i),
        ).images[0]
        base, ext = os.path.splitext(args.output)
        out_path = f"{base}_{i:02d}{ext}"
        img.save(out_path)
        print(f"  Saved to {out_path}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
