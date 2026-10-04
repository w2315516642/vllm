# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gather_zero_kernel(
    pool,
    ids,
    valid,
    out,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    VALID_STRIDE: tl.constexpr,
    ID_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(ids + row * ID_STRIDE).to(tl.int64)
    enabled = tl.load(valid + row * VALID_STRIDE)
    offset = slot * S0 + (x // (V * K)) * S1 + ((x // K) % V) * S2 + (x % K) * S3
    value = tl.load(pool + offset, mask=(x < H * V * K) & enabled, other=0)
    tl.store(out + row * H * V * K + x, value, mask=x < H * V * K)


@triton.jit
def _scatter_cast_kernel(
    src,
    ids,
    pool,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    R2: tl.constexpr,
    R3: tl.constexpr,
    ID_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(ids + row * ID_STRIDE).to(tl.int64)
    offset = slot * S0 + (x // (V * K)) * S1 + ((x // K) % V) * S2 + (x % K) * S3
    src_offset = row * R0 + (x // (V * K)) * R1 + ((x // K) % V) * R2 + (x % K) * R3
    value = tl.load(src + src_offset, mask=x < H * V * K, other=0)
    tl.store(pool + offset, value, mask=x < H * V * K)


def gather_gdn_state(
    pool: torch.Tensor, ids: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Gather [slot, H, V, K] states, zeroing rows without initial state."""
    h, v, k = pool.shape[1:]
    out = torch.empty((ids.numel(), h, v, k), device=pool.device, dtype=pool.dtype)
    _gather_zero_kernel[(ids.numel(), triton.cdiv(h * v * k, 1024))](
        pool,
        ids,
        valid,
        out,
        h,
        v,
        k,
        *pool.stride(),
        valid.stride(0),
        ids.stride(0),
        BLOCK=1024,
    )
    return out


def scatter_gdn_state(pool: torch.Tensor, ids: torch.Tensor, src: torch.Tensor) -> None:
    """Cast and store [N, H, V, K] states at unique, valid slot indices."""
    h, v, k = pool.shape[1:]
    _scatter_cast_kernel[(ids.numel(), triton.cdiv(h * v * k, 1024))](
        src,
        ids,
        pool,
        h,
        v,
        k,
        *pool.stride(),
        *src.stride(),
        ids.stride(0),
        BLOCK=1024,
    )
