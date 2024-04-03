# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
import json
import os

from collections import namedtuple
import einops

import torch
import torch.nn as nn

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.modules.mamba_simple_diffusion import Block_with_Skip
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

from libs.timm import trunc_normal_

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)

    args = timesteps[:, None].float() * freqs[None]

    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.to(dtype=timesteps.dtype)


def create_block(
    d_model,
    ssm_cfg=None,
    skip=False,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block_with_Skip(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        skip=skip,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        factory_kwargs=factory_kwargs,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, patch_size, in_chans, embed_dim, factory_kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            **factory_kwargs,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(
        x, "B (h w) (p1 p2 C) -> B C (h p1) (w p2)", h=h, p1=patch_size, p2=patch_size
    )
    return x


class Mamba_Diffusion_UNet(nn.Module):
    def __init__(
        self,
        d_in_chans: int,
        d_model: int,
        n_layer: int,
        img_dim: int,
        patch_size: int = 1,
        n_context_token: int = 0,
        d_context: int = 0,
        ssm_cfg=None,
        use_pe=True,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        self.factory_kwargs = factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.

        self.fused_add_norm = fused_add_norm
        print("fused_add_norm", fused_add_norm)
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.embed_dim = d_model
        self.context_dim = d_context
        self.n_token = img_dim * img_dim // patch_size**2
        self.d_in_chans = d_in_chans
        self.n_context_token = n_context_token
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=d_in_chans,
            embed_dim=self.embed_dim,
            factory_kwargs=factory_kwargs,
        )
        self.time_embed = nn.Sequential(
            nn.Linear(self.embed_dim, 4 * self.embed_dim, **factory_kwargs),
            nn.SiLU(),
            nn.Linear(4 * self.embed_dim, self.embed_dim, **factory_kwargs),
        )
        if self.n_context_token > 0:
            self.context_embed = nn.Linear(
                self.context_dim, self.embed_dim, **factory_kwargs
            )
        print("n_context_token", self.n_context_token)
        self.extras = 1 + self.n_context_token
        self.use_pe = use_pe
        if self.use_pe:
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, self.extras + self.n_token, self.embed_dim, **factory_kwargs
                )
            )
            trunc_normal_(self.pos_embed, std=0.02)
        print("use_pe", self.use_pe)

        self.in_blocks = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    skip=False,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer // 2)
            ]
        )
        self.mid_block = create_block(
            d_model,
            ssm_cfg=ssm_cfg,
            skip=False,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=n_layer // 2,
            **factory_kwargs,
        )

        self.out_blocks = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    skip=True,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=n_layer // 2 + i + 1,
                    **factory_kwargs,
                )
                for i in range(n_layer // 2)
            ]
        )

        self.patch_dim = self.patch_size**2 * self.d_in_chans
        self.decoder_pred = nn.Linear(
            self.embed_dim, self.patch_dim, bias=True, **factory_kwargs
        )
        self.final_layer = nn.Conv2d(
            self.d_in_chans, self.d_in_chans, 3, padding=1, **factory_kwargs
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states, timesteps, context=None, inference_params=None):
        hidden_states = hidden_states.to(**self.factory_kwargs)
        timesteps = timesteps.to(**self.factory_kwargs)
        hidden_states = self.patch_embed(hidden_states)
        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        if self.n_context_token > 0:
            context_token = self.context_embed(context.to(**self.factory_kwargs))
            hidden_states = torch.cat((time_token, context_token, hidden_states), dim=1)
        else:
            hidden_states = torch.cat((time_token, hidden_states), dim=1)
        if self.use_pe:
            hidden_states = hidden_states + self.pos_embed

        residual = None
        skips = []
        for blk in self.in_blocks:
            hidden_states, residual = blk(
                hidden_states, residual, inference_params=inference_params
            )
            skips.append(hidden_states)

        hidden_states, residual = self.mid_block(
            hidden_states, residual, inference_params=inference_params
        )

        for blk in self.out_blocks:
            hidden_states, residual = blk(
                hidden_states, skips.pop(), residual, inference_params=inference_params
            )

        # not use this anymore, we don't add LayerNorm in the end.
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # dongzhuoyao, not use this
            raise NotImplementedError
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        hidden_states = self.decoder_pred(hidden_states)
        hidden_states = hidden_states[:, self.extras :, :]
        hidden_states = unpatchify(hidden_states, self.d_in_chans)
        hidden_states = self.final_layer(hidden_states)
        return hidden_states
