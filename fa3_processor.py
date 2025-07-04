"""
Adapted from 
https://github.com/huggingface/flux-fast/blob/156281514e2725782ffab9431d4004840f7e3b4d/utils/pipeline_utils.py#L87
"""

import torch
from typing import List, Optional
import inspect


@torch.library.custom_op("flash::flash_attn_func", mutates_args=())
def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    # probably wrong type for these 4
    qv: Optional[float] = None,
    q_descale: Optional[float] = None,
    k_descale: Optional[float] = None,
    v_descale: Optional[float] = None,
    window_size: Optional[List[int]] = None,
    sink_token_length: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    # probably wrong type for this too
    pack_gqa: Optional[float] = None,
    deterministic: bool = False,
    sm_margin: int = 0,
) -> torch.Tensor:  # Tuple[torch.Tensor, torch.Tensor]:
    if window_size is None:
        window_size = (-1, -1)
    else:
        window_size = tuple(window_size)

    import flash_attn_interface

    dtype = torch.float8_e4m3fn

    sig = inspect.signature(flash_attn_interface.flash_attn_func)
    accepted = set(sig.parameters)
    all_kwargs = {
        "softmax_scale": softmax_scale,
        "causal": causal,
        "qv": qv,
        "q_descale": q_descale,
        "k_descale": k_descale,
        "v_descale": v_descale,
        "window_size": window_size,
        "sink_token_length": sink_token_length,
        "softcap": softcap,
        "num_splits": num_splits,
        "pack_gqa": pack_gqa,
        "deterministic": deterministic,
        "sm_margin": sm_margin,
    }
    kwargs = {k: v for k, v in all_kwargs.items() if k in accepted}

    outputs = flash_attn_interface.flash_attn_func(
        q.to(dtype),
        k.to(dtype),
        v.to(dtype),
        **kwargs,
    )
    return outputs[0]


@flash_attn_func.register_fake
def _(q, k, v, **kwargs):
    # two outputs:
    # 1. output: (batch, seq_len, num_heads, head_dim)
    # 2. softmax_lse: (batch, num_heads, seq_len) with dtype=torch.float32
    meta_q = torch.empty_like(q).contiguous()
    return meta_q  # , q.new_empty((q.size(0), q.size(2), q.size(1)), dtype=torch.float32)


# Copied FusedFluxAttnProcessor2_0 but using flash v3 instead of SDPA
class FlashFluxAttnProcessor3_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        try:
            import flash_attn_interface
        except ImportError:
            raise ImportError("flash_attention v3 package is required to be installed")

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # NB: transposes are necessary to match expected SDPA input shape
        hidden_states = flash_attn_func(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))[
            0
        ].transpose(1, 2)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
