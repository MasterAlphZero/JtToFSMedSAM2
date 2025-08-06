# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Optional, Tuple

import numpy as np

import jittor as jt
from jittor import nn

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention Is All You Need paper, generalized to work on images.
    """

    def __init__(
            self,
            num_pos_feats,
            temperature: int = 10000,
            normalize: bool = True,
            scale: Optional[float] = None,
            # Following settings only relevant
            # for warmping up cache for compilation
            warmup_cache: bool = True,
            image_size: int = 1024,
            strides: Tuple[int] = (4, 8, 16, 32),
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}
        if warmup_cache and jt.compiler.has_cuda:
            # Warmup cache for cuda, to help with compilation
            device = "cuda"
            for stride in strides:
                cache_key = (image_size // stride, image_size // stride)
                self._pe(1, *cache_key)

    def _encode_xy(self, x, y):
        # The positions are expected to be normalized
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = jt.arange(self.num_pos_feats, dtype=jt.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = jt.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos_y = jt.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)
        return pos_x, pos_y

    @jt.no_grad()
    def encode_boxes(self, x, y, w, h):
        pos_x, pos_y = self._encode_xy(x, y)
        pos = jt.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos

    encode = encode_boxes  # Backwards compatibility

    @jt.no_grad()
    def encode_points(self, x, y, labels):
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = jt.concat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos

    @jt.no_grad()
    def _pe(self, B, *cache_key):
        H, W = cache_key
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(B, 1, 1, 1)

        y_embed = (
            jt.arange(1, H + 1, dtype=jt.float32)
            .view(1, -1, 1)
            .repeat(B, 1, W)
        )
        x_embed = (
            jt.arange(1, W + 1, dtype=jt.float32)
            .view(1, 1, -1)
            .repeat(B, H, 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = jt.arange(self.num_pos_feats, dtype=jt.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = jt.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = jt.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = jt.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos

    @jt.no_grad()
    def execute(self, x: jt.Var):
        B = x.shape[0]
        cache_key = (x.shape[-2], x.shape[-1])
        return self._pe(B, *cache_key)


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * jt.randn(2, num_pos_feats),
        )

    def _pe_encoding(self, coords: jt.Var) -> jt.Var:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return jt.concat([jt.sin(coords), jt.cos(coords)], dim=-1)

    def execute(self, size: Tuple[int, int]) -> jt.Var:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        grid = jt.ones((h, w))
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(jt.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
            self, coords_input: jt.Var, image_size: Tuple[int, int]
    ) -> jt.Var:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.float())  # B x N x C


# Rotary Positional Encoding, adapted from:
# 1. https://github.com/meta-llama/codellama/blob/main/llama/model.py
# 2. https://github.com/naver-ai/rope-vit
# 3. https://github.com/lucidrains/rotary-embedding-torch


def init_t_xy(end_x: int, end_y: int):
    t = jt.arange(end_x * end_y, dtype=jt.float32)
    t_x = (t % end_x).float()
    t_y = (t // end_x).float()
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    freqs_x = 1.0 / (theta ** (jt.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (jt.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = jt.multiply(t_x.unsqueeze(1), freqs_x.unsqueeze(0))
    freqs_y = jt.multiply(t_y.unsqueeze(1), freqs_y.unsqueeze(0))

    # 计算 cos 和 sin
    cos_x = jt.cos(freqs_x)
    sin_x = jt.sin(freqs_x)
    cos_y = jt.cos(freqs_y)
    sin_y = jt.sin(freqs_y)

    # 拼接 x、y 方向的结果；得到 cos 和 sin 矩阵，形状均为 [end_x*end_y, dim//2]
    cos = jt.concat([cos_x, cos_y], dim=-1)
    sin = jt.concat([sin_x, sin_y], dim=-1)
    # 将 cos 和 sin 合并到最后一个维度：形状 [end_x*end_y, dim//2, 2]
    freqs_cis = jt.stack([cos, sin], dim=-1)
    return freqs_cis


# def reshape_for_broadcast(freqs_cis: jt.Var, x: jt.Var):
#     ndim = x.ndim
#     assert 0 <= 1 < ndim
#     assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
#     shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
#     return freqs_cis.view(*shape)


def apply_rotary_enc(
        xq: jt.Var,
        xk: jt.Var,
        freqs_cis: jt.Var,
        repeat_freqs_k: bool = False,
):
    """
    对查询 xq 和键 xk 应用旋转位置编码 freqs_cis（[seq, dim//2, 2]，最后一维为 [cos, sin]）。
    返回编码后的 (xq_out, xk_out)。支持可选的对 K 重复频率（repeat_freqs_k）。
    """
    # 提取 cos 和 sin，形状均为 [seq, dim//2]
    cos = freqs_cis[..., 0]  # 实部
    sin = freqs_cis[..., 1]  # 虚部

    # 准备广播到 [batch, heads, seq, dim//2]
    # 新形状 [1, 1, seq, dim//2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Q 的偶数/奇数位分离
    xq_even = xq[..., 0::2]
    xq_odd = xq[..., 1::2]
    # 旋转编码公式： (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos)
    xq_out_even = xq_even * cos - xq_odd * sin
    xq_out_odd = xq_even * sin + xq_odd * cos
    # 重新 interleave 偶数/奇数位回原始形状
    xq_out = jt.stack([xq_out_even, xq_out_odd], dim=-1).flatten(-2)
    xq_out = xq_out.cast(xq.dtype)  # 恢复原始 dtype

    # 如果没有提供 xk 或长度为0，则只返回 q
    if xk is None or xk.shape[-2] == 0:
        return xq_out, xk

    # K 的旋转编码，可能需要重复频率矩阵
    if repeat_freqs_k:
        # 假设 xk_seq = r * xq_seq
        r = xk.shape[-2] // xq.shape[-2]
        cos = cos.repeat(1, 1, r, 1)
        sin = sin.repeat(1, 1, r, 1)

    xk_even = xk[..., 0::2]
    xk_odd = xk[..., 1::2]
    xk_out_even = xk_even * cos - xk_odd * sin
    xk_out_odd = xk_even * sin + xk_odd * cos
    xk_out = jt.stack([xk_out_even, xk_out_odd], dim=-1).flatten(-2)
    xk_out = xk_out.cast(xk.dtype)

    return xq_out, xk_out