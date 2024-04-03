# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
import json
import os

from collections import namedtuple
import einops

import torch
import torch.nn as nn


from mamba_ssm.modules.mamba_simple import Mamba


try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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


class Diff_SSM_Block(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        mamba_kwargs,
        downsacale_ratio,
        need_modulate=True,
        skip=False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.need_modulate = need_modulate
        if need_modulate:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True)
            )

        self.mamba = Mamba(hidden_size, **mamba_kwargs)
        self.hg_dense = HourglassDenseLayer(
            hidden_size, downsacale_ratio=downsacale_ratio
        )
        self.hg_fuse = HourglassFusionLayer(
            hidden_size, downscale_ratio=downsacale_ratio
        )

        self.skip_linear = nn.Linear(2 * hidden_size, hidden_size) if skip else None

    def forward(self, x, c=None, skip=None):
        if self.skip_linear is not None:
            hidden_states = self.skip_linear(torch.cat([hidden_states, skip], dim=-1))
        if self.need_modulate:
            raise NotImplementedError
            (
                shift_msa,
                scale_msa,
                scale_alpha,
            ) = self.adaLN_modulation(
                c
            ).chunk(3, dim=1)

            x1 = modulate(self.norm(x), shift_msa, scale_msa)
            x1_2 = self.mamba(self.hg_dense(x1))
            x1_1 = scale_alpha.unsqueeze(1) * self.hg_fuse(x1_2, x1)
            x = x + x1_1
        else:
            x1 = self.norm(x)
            x1_2 = self.mamba(self.hg_dense(x1))
            x1_1 = self.hg_fuse(x1_2, x1)
            x = x + x1_1
        return x


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


class Mamba_Diffusion_UViT(nn.Module):
    def __init__(
        self,
        d_in_chans: int,
        d_model: int,
        n_layer: int,
        img_dim: int,
        patch_size: int = 1,
        n_context_token: int = 0,
        d_context: int = 0,
        downscale_ratio: int = 2,
        ssm_cfg=dict(),
        use_pe=True,
        norm_epsilon: float = 1e-6,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.factory_kwargs = factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.

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
            from libs.timm import trunc_normal_

            trunc_normal_(self.pos_embed, std=0.02)
        print("use_pe", self.use_pe)

        mamba_kwargs = dict(**factory_kwargs)
        mamba_kwargs.update(ssm_cfg)

        self.in_blocks = nn.ModuleList(
            [
                Diff_SSM_Block(
                    hidden_size=d_model,
                    downsacale_ratio=downscale_ratio,
                    mamba_kwargs=mamba_kwargs,
                    need_modulate=False,
                )
                .to(device)
                .to(dtype)
                for i in range(n_layer // 2)
            ]
        )
        self.mid_block = (
            Diff_SSM_Block(
                hidden_size=d_model,
                downsacale_ratio=downscale_ratio,
                mamba_kwargs=mamba_kwargs,
                need_modulate=False,
            )
            .to(device)
            .to(dtype)
        )

        self.out_blocks = nn.ModuleList(
            [
                Diff_SSM_Block(
                    hidden_size=d_model,
                    downsacale_ratio=downscale_ratio,
                    mamba_kwargs=mamba_kwargs,
                    need_modulate=False,
                )
                .to(device)
                .to(dtype)
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

        self.norm_f = (nn.LayerNorm)(d_model, eps=norm_epsilon, **factory_kwargs)

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def forward(self, hidden_states, timesteps, context=None):
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

        skips = []
        for blk in self.in_blocks:
            hidden_states = blk(hidden_states)
            skips.append(hidden_states)

        hidden_states = self.mid_block(hidden_states)

        for blk in self.out_blocks:
            hidden_states = blk(hidden_states, skip=skips.pop())

        hidden_states = self.norm_f(hidden_states.to(dtype=self.norm_f.weight.dtype))

        hidden_states = self.decoder_pred(hidden_states)
        hidden_states = hidden_states[:, self.extras :, :]
        hidden_states = unpatchify(hidden_states, self.d_in_chans)
        hidden_states = self.final_layer(hidden_states)
        return hidden_states


if __name__ == "__main__":
    model = Mamba_Diffusion_UViT(
        d_in_chans=4,
        d_model=128,
        n_layer=4,
        img_dim=32,
        use_pe=False,
        d_context=768,
        n_context_token=77,
        device="cuda",
        dtype=torch.float32,
    )
    # print(f"param num: {cnt_params(model)}")
    x = torch.randn(4, 4, 32, 32).to("cuda")
    t = torch.randn(4).to("cuda")
    context = torch.randn(4, 77, 768).to("cuda")
    out = model(x, t, context)
    print(out.shape)
