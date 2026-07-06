import os
import sys
import argparse
import numpy as np
import torch

torch.manual_seed(42)

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

from diffusers import DiffusionPipeline
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

logger = logging.get_logger(__name__)

MODEL_PATH_DEFAULT = (
    "C:\\Users\\Emanuel\\.cache\\huggingface\\hub\\"
    "models--Freepik--flux.1-lite-8B-alpha\\snapshots\\"
    "812d376439b6e37b0e6f6dd401b2a98b1effacdb"
)


def teacache_forward(
    self,
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
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
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
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    if self.enable_teacache:
        inp = hidden_states.clone()
        temb_ = temb.clone()
        modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)
        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(
                ((modulated_inp - self.previous_modulated_input).abs().mean()
                 / self.previous_modulated_input.abs().mean()).cpu().item()
            )
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0

    if self.enable_teacache:
        if not should_calc:
            hidden_states += self.previous_residual
        else:
            ori_hidden_states = hidden_states.clone()
            for index_block, block in enumerate(self.transformer_blocks):
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
                if controlnet_block_samples is not None:
                    interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    if controlnet_blocks_repeat:
                        hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    else:
                        hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            for index_block, block in enumerate(self.single_transformer_blocks):
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
                if controlnet_single_block_samples is not None:
                    interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states[:, encoder_hidden_states.shape[1]:, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1]:, ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                    )

            hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]
            self.previous_residual = hidden_states - ori_hidden_states
    else:
        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                if controlnet_blocks_repeat:
                    hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1]:, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1]:, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

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
        description="FLUX TeaCache Inference with local model"
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
        default="output_teacache.png",
        help="Caminho para salvar a imagem gerada",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.6,
        help="TeaCache relative L1 threshold",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=28,
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
        print(f"ERRO: Nao foi possivel carregar o modelo local em:\n  {args.model_path}")
        print(f"Detalhes: {e}")
        sys.exit(1)

    FluxTransformer2DModel.forward = teacache_forward

    pipeline.transformer.__class__.enable_teacache = True
    pipeline.transformer.__class__.cnt = 0
    pipeline.transformer.__class__.num_steps = args.steps
    pipeline.transformer.__class__.rel_l1_thresh = args.thresh
    pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
    pipeline.transformer.__class__.previous_modulated_input = None
    pipeline.transformer.__class__.previous_residual = None

    pipeline.to("cuda")
    pipeline.vae.enable_slicing()
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
