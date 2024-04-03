import torch
import torch.nn as nn
import numpy as np
import math

import math
from functools import partial
import json
import os

from collections import namedtuple
import einops

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba_simple import Mamba
import xformers.ops
from timm.models.vision_transformer import Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class HourglassDenseLayer(nn.Module):
    def __init__(self, input_dim, downsacale_ratio):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // downsacale_ratio, bias=True),
            nn.SiLU(),
            nn.Linear(input_dim // downsacale_ratio, input_dim, bias=True),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class HourglassFusionLayer(nn.Module):
    def __init__(self, input_dim, downscale_ratio):
        super().__init__()
        self.mlp_main = nn.Sequential(
            nn.Linear(input_dim, input_dim // downscale_ratio, bias=True),
            nn.SiLU(),
            # nn.Linear(input_dim // downscale_ratio, input_dim, bias=True),
        )
        self.mlp_residual = nn.Sequential(
            nn.Linear(input_dim, input_dim // downscale_ratio, bias=True),
            nn.SiLU(),
            # nn.Linear(input_dim // downscale_ratio, input_dim, bias=True),
        )
        self.mlp_fuse = nn.Sequential(
            nn.Linear(input_dim // downscale_ratio, input_dim, bias=True),
            # nn.SiLU(),
            # nn.Linear(input_dim // downscale_ratio, input_dim, bias=True),
        )

    def forward(self, x, residual):
        x = self.mlp_main(x)
        residual = self.mlp_residual(residual)
        x = self.mlp_fuse(x * residual)
        return x


class BiGSLayer(
    nn.Module
):  # https://github.com/jxiw/BiGS/blob/d630e392c7b51d449d576edbe2defadf094ab00f/BiGS/modeling_bigs.py#L319
    def __init__(
        self,
        hidden_size,
        downsacale_ratio,
        layer_norm_eps=1e-6,
        **mamba_kwargs,
    ):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        intermediate_size = hidden_size // downsacale_ratio

        # ssm layers
        self.fs4 = Mamba(hidden_size, **mamba_kwargs)
        self.bs4 = Mamba(hidden_size, **mamba_kwargs)
        # dense layers
        self.dv = nn.Linear(hidden_size, intermediate_size)
        self.du_forward = nn.Linear(hidden_size, hidden_size)
        self.du_backward = nn.Linear(hidden_size, hidden_size)
        self.duc_forward = nn.Linear(hidden_size, hidden_size)
        self.duc_backward = nn.Linear(hidden_size, hidden_size)
        self.dol = nn.Linear(hidden_size, intermediate_size)
        self.do = nn.Linear(intermediate_size, hidden_size)

    def __call__(self, hidden_states):
        hidden_residual = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        # gating
        v = nn.functional.gelu(self.dv(hidden_states))
        ###
        u_forward = nn.functional.gelu(self.du_forward(hidden_states))
        u_backward = nn.functional.gelu(
            self.du_backward(torch.flip(hidden_states, dims=[1]))
        )
        # s4 layers
        fs4_output = self.fs4(u_forward)
        bs4_output = self.bs4(u_backward)
        # instead of sum, we use multiplication
        uc_forward = self.duc_forward(fs4_output)
        uc_backward = torch.flip(self.duc_backward(bs4_output), dims=[1])
        hidden_states = self.do(
            nn.functional.gelu(self.dol(uc_forward * uc_backward)) * v
        )
        hidden_states = hidden_residual + hidden_states
        return hidden_states


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        proj_drop=0.0,
    ):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape
        context = context.to(x.device)

        #########
        q = self.q_linear(x)  # .view(B, self.num_heads, -1, self.head_dim)
        kv = self.kv_linear(context)  # .view(B, 2, self.num_heads, -1, self.head_dim)
        q = einops.rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
        kv = einops.rearrange(kv, "B L (K H D) -> K B H L D", K=2, H=self.num_heads)
        k, v = kv.unbind(0)
        # q,k,v = B, H, L, D
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = einops.rearrange(x, "B H L D -> B L (H D)")
        #########
        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        uncond_prob,
        act_layer=nn.GELU(approximate="tanh"),
        token_num=120,
    ):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            drop=0,
        )
        self.register_buffer(
            "y_embedding",
            nn.Parameter(torch.randn(token_num, in_channels) / in_channels**0.5),
        )
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        if train:
            assert (
                caption.shape[1:] == self.y_embedding.shape
            ), f"{caption.shape} is not {self.y_embedding.shape}"
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


class Diff_SSM_Block(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        mamba_kwargs,
        downsacale_ratio,
        is_bidirectional=False,
        has_text=False,
        num_heads=None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        if not is_bidirectional:
            self.mamba = Mamba(hidden_size, **mamba_kwargs)
        else:
            self.mamba = BiGSLayer(
                hidden_size, downsacale_ratio=downsacale_ratio, **mamba_kwargs
            )
        self.has_text = has_text
        if has_text:
            self.cross_attn = MultiHeadCrossAttention(
                hidden_size,
                num_heads,
            )

        self.hg_dense = HourglassDenseLayer(
            hidden_size, downsacale_ratio=downsacale_ratio
        )
        self.hg_fuse = HourglassFusionLayer(
            hidden_size, downscale_ratio=downsacale_ratio
        )

    def forward(self, x, t=None, context=None, mask=None):
        (
            shift_msa,
            scale_msa,
            scale_alpha,
        ) = self.adaLN_modulation(
            t
        ).chunk(3, dim=1)

        x1 = modulate(self.norm(x), shift_msa, scale_msa)
        x1_2 = self.mamba(self.hg_dense(x1))

        if self.has_text:
            x1_2 = x1_2 + self.cross_attn(x1_2, context, mask)
            x1_2 = self.norm2(x1_2)  # Tao added by Jan.28

        x1_1 = scale_alpha.unsqueeze(1) * self.hg_fuse(x1_2, x1)
        x = x + x1_1

        return x
