import torch
import types
import numpy as np
from typing import Any, Dict, Optional, Union
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from dpcache import CacheHelper, init_cache

logger = logging.get_logger(__name__)


def dpcache_flux_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

    if joint_attention_kwargs is None:
        joint_attention_kwargs = {}

    # [Cache] Init cache helper & current
    helper = CacheHelper(self.cache_data)
    current = helper.init_current(self, timestep)

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective.")

    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None

    temb = self.time_text_embed(timestep, pooled_projections) if guidance is None else self.time_text_embed(timestep, guidance, pooled_projections)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning("Passing `txt_ids` 3d torch.Tensor is deprecated." "Please remove the batch dimension and pass it as a 2d torch Tensor")
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning("Passing `img_ids` 3d torch.Tensor is deprecated." "Please remove the batch dimension and pass it as a 2d torch Tensor")
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    # Double transformer blocks
    # [Cache] Update stream
    helper.update_stream("double_stream", len(self.transformer_blocks))

    for index_block, block in enumerate(self.transformer_blocks):
        # [Cache] Update layer
        helper.update_layer(index_block)
        # [Cache] pass cache helper to block
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            cache_helper=helper,
        )

        # controlnet residual
        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
            interval_control = int(np.ceil(interval_control))
            if controlnet_blocks_repeat:
                hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
            else:
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    # Single transformer blocks
    # [Cache] Update stream
    helper.update_stream("single_stream", len(self.single_transformer_blocks))

    for index_block, block in enumerate(self.single_transformer_blocks):
        # [Cache] Update layer
        helper.update_layer(index_block)
        # [Cache] pass cache helper to block
        hidden_states = block(
            hidden_states=hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            cache_helper=helper,
        )

        # controlnet residual
        if controlnet_single_block_samples is not None:
            interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
            interval_control = int(np.ceil(interval_control))
            hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] + controlnet_single_block_samples[index_block // interval_control]
            )

    # [Cache] update encoder_seq_len (flux only)
    encoder_seq_len = encoder_hidden_states.shape[1]
    hidden_states = hidden_states[:, encoder_seq_len:, ...]

    current["encoder_seq_len"] = encoder_seq_len

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    # [Cache] Cali after single_stream
    if helper.should_perform_calibration(cali_stream="single_stream"):
        cali_prefix = getattr(self, "cali_prefix", "flux_dpcache")
        helper.perform_calibration(cali_stream="single_stream", cali_prefix=cali_prefix)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


def dpcache_flux_double_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: torch.Tensor,
    cache_helper: CacheHelper,
) -> torch.Tensor:
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)

    # [Cache] full compute
    if cache_helper.should_compute():
        cache_helper.print_schedule_status(action="compute")

        # Attention
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs
            raise NotImplementedError("IP adapter not implemented for DPCache yet.")

        # Process attention outputs for hidden_states (img)
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        # MLP for hidden_states (img)
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        # Process attention outputs for encoder_hidden_states (txt)
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        # MLP for encoder_hidden_states (txt)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        # combined_output is cached for double_stream
        # [Cache] cache combined_output
        if cache_helper.should_cache_feature():
            combined_output = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            cache_helper.cache_feature(combined_output, module_name="block_output_hidden_states")

    # [Cache] use Taylor-DP cache
    elif cache_helper.should_skip_with_cache():
        cache_helper.print_schedule_status(action="skip")
        if cache_helper.is_last_layer():
            combined_output = cache_helper.retrieve_cached_feature(module_name="block_output_hidden_states")
            # split encoder_hidden_states and hidden_states
            encoder_seq_len = encoder_hidden_states.shape[1]
            encoder_hidden_states = combined_output[:, :encoder_seq_len, :]
            hidden_states = combined_output[:, encoder_seq_len:, :]

    else:
        raise ValueError(f"Not supported type: {cache_helper.current['type']}")

    return cache_helper.clip_fp16(encoder_hidden_states), hidden_states


def dpcache_flux_single_block_forward(
    self,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: torch.Tensor,
    cache_helper: CacheHelper,
) -> torch.Tensor:
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    gate = gate.unsqueeze(1)
    residual = hidden_states

    # [Cache] full compute
    if cache_helper.should_compute():
        cache_helper.print_schedule_status(action="compute")

        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = gate * hidden_states
        hidden_states = residual + hidden_states

        # [Cache] cache hidden_states for inference or save history for calibration
        if cache_helper.should_cache_feature():
            cache_helper.cache_feature(hidden_states, module_name="block_output_hidden_states")
        elif cache_helper.should_save_history():

            def preprocess(tensor):
                encoder_seq_len = cache_helper.current.get("encoder_seq_len")
                if encoder_seq_len is not None:
                    return tensor[:, encoder_seq_len:, ...]
                return tensor

            cache_helper.save_feature_history(hidden_states, module_name="block_output_hidden_states", preprocess_fn=preprocess)

    # [Cache] use Taylor-DP cache
    elif cache_helper.should_skip_with_cache():
        cache_helper.print_schedule_status(action="skip")
        if cache_helper.is_last_layer():
            hidden_states = cache_helper.retrieve_cached_feature(module_name="block_output_hidden_states")

    else:
        raise ValueError(f"Not supported type: {cache_helper.current['type']}")

    return cache_helper.clip_fp16(hidden_states)


def apply_dpcache_to_flux(
    flux_pipeline,
    mode="Taylor-DP",
    cali=False,
    num_steps=50,
    first_full_steps=5,
    last_full_steps=0,
    order=2,
    k=9,
    cost_matrix_path=None,
    cali_prefix="flux_dpcache",
    cost_metric="l1",
):
    flux_pipeline.transformer.cache_data = init_cache(
        flux_pipeline.transformer,
        mode=mode,
        num_steps=num_steps,
        first_full_steps=first_full_steps,
        last_full_steps=last_full_steps,
        order=order,
        selected_steps=None,
        k=k,
        cali=cali,
        cost_matrix_path=cost_matrix_path or "flux_cost_matrix.pkl",
    )

    if mode in ("Taylor-DP"):
        flux_pipeline.transformer.forward = types.MethodType(dpcache_flux_forward, flux_pipeline.transformer)

        for block in flux_pipeline.transformer.transformer_blocks:
            block.forward = types.MethodType(dpcache_flux_double_block_forward, block)

        for block in flux_pipeline.transformer.single_transformer_blocks:
            block.forward = types.MethodType(dpcache_flux_single_block_forward, block)

    if cali:
        flux_pipeline.transformer.__class__.cali_prefix = cali_prefix
        flux_pipeline.transformer.__class__.cost_metric = cost_metric

    return flux_pipeline
