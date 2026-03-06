import torch
import types
import bisect
import pickle
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


def dpcache_wan_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    # [Cache] Init cache helper & current
    current_stream = attention_kwargs.get("stream", "cond_stream")
    helper = CacheHelper(self.cache_data)
    if current_stream == "cond_stream":
        current = helper.init_current(self, timestep)


    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    rotary_emb = self.rope(hidden_states)

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    # [Cache] Update stream
    helper.update_stream(current_stream, len(self.blocks))

    # 4. Transformer blocks
    for i, block in enumerate(self.blocks):
        # [Cache] Update layer
        helper.update_layer(i)
        # [Cache] pass cache helper to block
        hidden_states = block(
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            rotary_emb,
            cache_helper=helper
        )

    # 5. Output norm, projection & unpatchify
    shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    # hidden_states = get_sp_group().all_gather(hidden_states, dim=1)

    hidden_states = hidden_states.reshape(
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        p_t,
        p_h,
        p_w,
        -1,
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    
    # [Cache] Cali after blocks
    if helper.should_perform_calibration(cali_stream="cond_stream"):
        cali_prefix = getattr(self, "cali_prefix", "wan_dpcache")
        helper.perform_calibration(cali_stream="cond_stream", cali_prefix=cali_prefix)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


def dpcache_wan_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    rotary_emb: torch.Tensor,
    cache_helper: Optional[CacheHelper] = None
) -> torch.Tensor:
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (self.scale_shift_table + temb.float()).chunk(6, dim=1)
    # [Cache] full compute
    if cache_helper.should_compute():
        cache_helper.print_schedule_status(action="compute")

        # self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = hidden_states + attn_output

        # ffn
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        
        # [Cache] cache hidden_states for inference or save history for calibration
        if cache_helper.should_cache_feature():
            cache_helper.cache_feature(hidden_states, module_name="block_output_hidden_states")
        elif cache_helper.should_save_history():
            cache_helper.save_feature_history(hidden_states, module_name="block_output_hidden_states")

    # [Cache] use Taylor-DP cache
    elif cache_helper.should_skip_with_cache():
        cache_helper.print_schedule_status(action="skip")
        if cache_helper.is_last_layer():
            hidden_states = cache_helper.retrieve_cached_feature(module_name="block_output_hidden_states")

    else:
        raise ValueError(f"Not supported type: {cache_helper.current['type']}")

    return hidden_states


def apply_dpcache_to_wan(
    pipe,
    mode="Taylor-DP",
    cali=False,
    num_steps=30,
    first_full_steps=4,
    last_full_steps=1,
    order=2,
    k=10,
    cost_matrix_path=None,
    cali_prefix="wan_cfg_3_720p",
):
    pipe.transformer.cache_data = init_cache(
        pipe.transformer,
        mode=mode,
        model_name="wan",
        num_steps=num_steps,
        first_full_steps=first_full_steps,
        last_full_steps=last_full_steps,
        order=order,
        selected_steps=None,
        k=k,
        cali=cali,
        cost_matrix_path=cost_matrix_path or "wan_cost_matrix.pkl",
    )
    
    if mode in ("Taylor-DP"):
        for block in pipe.transformer.blocks:
            block.forward = types.MethodType(dpcache_wan_block_forward, block)
    
        pipe.transformer.forward = types.MethodType(dpcache_wan_forward, pipe.transformer)
    
    if cali:
        pipe.transformer.__class__.cali_prefix = cali_prefix

    return pipe
