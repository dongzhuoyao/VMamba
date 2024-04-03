# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from mamba_ssm.models.block_diffssm import Diff_SSM_Block, CaptionEmbedder
import math
from functools import partial
import json
import os

from collections import namedtuple
import einops

import torch
import torch.nn as nn

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


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
        self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, **block_kwargs
    ):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape
        cond = cond.to(x.device)
        if False:
            q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
            kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
            k, v = kv.unbind(2)
            attn_bias = None
            if mask is not None:
                attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens(
                    [N] * B, mask
                )
            x = xformers.ops.memory_efficient_attention(
                q, k, v, p=self.attn_drop.p, attn_bias=attn_bias
            )  # [B,L,H,D]
        else:
            q = self.q_linear(x)  # .view(B, self.num_heads, -1, self.head_dim)
            kv = self.kv_linear(cond)  # .view(B, 2, self.num_heads, -1, self.head_dim)
            q = einops.rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
            kv = einops.rearrange(kv, "B L (K H D) -> K B H L D", K=2, H=self.num_heads)
            k, v = kv.unbind(0)
            # q,k,v = B, H, L, D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, "B H L D -> B L (H D)")
        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # q = self.q_linear(x).reshape(B, -1, self.num_heads, self.head_dim)
        # kv = self.kv_linear(cond).reshape(B, -1, 2, self.num_heads, self.head_dim)
        # k, v = kv.unbind(2)
        # attn_bias = None
        # if mask is not None:
        #     attn_bias = torch.zeros([B * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
        #     attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float('-inf'))
        # x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        # x = x.contiguous().reshape(B, -1, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)

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
        x1_1 = scale_alpha.unsqueeze(1) * self.hg_fuse(x1_2, x1)
        x = x + x1_1

        return x


#################################################################################
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Mamba_Diffusion_DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        d_in_chans: int,
        d_model: int,
        n_layer: int,
        img_dim: int,
        patch_size: int = 1,
        downscale_ratio: int = 2,
        is_bid: bool = False,
        has_text: bool = False,
        n_context_token: int = 0,
        d_context: int = 0,
        ssm_cfg=dict(),
        use_pe: bool = True,
        class_dropout_prob=0.0,  # 0.1
        num_heads=8,
        device=None,
        dtype=None,
    ):
        self.factory_kwargs = factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_channels = d_in_chans
        self.out_channels = d_in_chans
        self.patch_size = patch_size
        self.d_model = d_model
        self.is_bidirectional = is_bid
        print("is_bidirectional", is_bid)

        self.x_embedder = (
            PatchEmbed(img_dim, patch_size, self.in_channels, self.d_model, bias=True)
            .to(device)
            .to(dtype)
        )
        self.t_embedder = TimestepEmbedder(self.d_model).to(device).to(dtype)

        num_patches = self.x_embedder.num_patches
        self.use_pe = use_pe
        if use_pe:
            # Will use fixed sin-cos embedding:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.d_model), requires_grad=False
            )
            # .to(device)
            # .to(dtype)

        print("use_pe", use_pe)
        self.n_layer = n_layer

        self.has_text = has_text
        print("has_text", has_text)
        if has_text:
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.y_embedder = (
                CaptionEmbedder(
                    in_channels=d_context,
                    hidden_size=d_model,
                    uncond_prob=class_dropout_prob,
                    act_layer=approx_gelu,
                    token_num=n_context_token,
                )
                .to(device)
                .to(dtype)
            )

        mamba_kwargs = dict(**factory_kwargs)
        mamba_kwargs.update(ssm_cfg)

        self.blocks = nn.ModuleList(
            [
                Diff_SSM_Block(
                    hidden_size=d_model,
                    downsacale_ratio=downscale_ratio,
                    is_bidirectional=is_bid,
                    mamba_kwargs=mamba_kwargs,
                    has_text=has_text,
                    num_heads=num_heads,
                )
                .to(device)
                .to(dtype)
                for _ in range(self.n_layer)
            ]
        )
        self.final_layer = (
            FinalLayer(self.d_model, patch_size, self.out_channels).to(device).to(dtype)
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if self.use_pe:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.has_text:
            # Initialize label embedding table:
            nn.init.normal_(self.y_embedder.y_embedding, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, context):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2
        if self.use_pe:
            x = x + self.pos_embed
        # assert t[0] <= 10
        t = self.t_embedder(t * 1000)  # (N, D)

        if self.has_text:
            context = self.y_embedder(context, self.training)  # (N, D)
        c = t
        # c = t + y  # (N, D)
        for block in self.blocks:
            x = block(x, c, context=context)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


if __name__ == "__main__":
    pass
