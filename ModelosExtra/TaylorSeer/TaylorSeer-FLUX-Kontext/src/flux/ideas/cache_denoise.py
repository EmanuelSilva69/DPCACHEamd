import torch
from ..model import Flux
from torch import Tensor
from ..modules.cache_functions import cache_init

def denoise_cache(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    # extra img tokens (sequence-wise)
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
    guidance: float = 4.0,
    **kwargs
):  
    model_kwargs = {
        'interval': kwargs.get('interval'),
        'max_order': kwargs.get('max_order'),
        'first_enhance': kwargs.get('first_enhance')
    }

    # init cache
    cache_dic, current = cache_init(timesteps, model_kwargs=model_kwargs)
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    current['step']=0
    current['num_steps'] = len(timesteps)-1
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        img_input = img
        img_input_ids = img_ids   
        current['t'] = t_curr

        if img_cond_seq is not None:
            assert (
                img_cond_seq_ids is not None
            ), "You need to provide either both or neither of the sequence conditioning"
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

        #print(t_curr)
        pred = model(
            img=img_input,
            img_ids=img_input_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            cache_dic=cache_dic,
            current=current,
            guidance=guidance_vec,
        )
        #print(img.shape)
        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]
        img = img + (t_prev - t_curr) * pred
        current['step'] += 1

    return img
