# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Reference block-sparsity counter for skip-softmax, in plain PyTorch.

The kernel does not report sparsity: counters would need atomics on the hot
path, so they were deliberately left out of the upstream change. This module
recomputes the same quantity independently, which is the better arrangement for
a reproduction bundle anyway. If a number can only be produced by the thing
being evaluated, it cannot corroborate it.

What is counted, matching ``process_attention_tile`` in the Gluon kernel:

  * Queries are tiled by ``BLOCK_M``, keys by ``BLOCK_N``. For a query tile
    starting at ``q_start``, the main loop visits ``q_start // BLOCK_N`` fully
    unmasked K/V blocks, then two causal boundary blocks on the diagonal. All
    of those count toward the denominator.
  * A block is *skipped* only when every one of the ``BLOCK_M`` rows skips it
    (the kernel's ``all_skip``); that is the case where the P@V matmul is
    elided, and it is the only case that saves time. Partially skipped blocks
    still run the matmul, so they are not sparsity, but they are not harmless
    either: the skipped rows are zeroed out of ``p``, so the output changes.
    Sparsity and "does the result differ from dense" are therefore different
    questions, and ``count_partial`` answers the second one.
  * A row skips a block when ``exp(row_max - m_i) < threshold``, where both are
    softmax-scaled scores, ``row_max`` is that row's maximum over the block and
    ``m_i`` is the running max from *before* the block. The kernel compares in
    log2 space; the algebra is identical, and this module compares in natural
    log space to keep it readable.
  * Skipped rows do not advance the running max, so a skip decision feeds back
    into later blocks. This is a sequential recurrence over blocks, not a
    one-shot mask, and it is why sparsity cannot be computed from the score
    matrix alone.

Sequences shorter than ``BLOCK_N`` take a separate single-tile path in the
kernel that has no skipping, and are excluded here for the same reason.

Validated against an instrumented build of the kernel that does carry atomic
counters: across five shapes (seqlens 512 to 2048, GQA ratios 1 to 4,
thresholds 0.9 to 4.0) both the visited-block and skipped-block counts matched
exactly, not approximately. ``scripts/verify_reference.py`` re-runs a weaker
form of that check, comparing against the exact output difference rather than
against counters, since the shipped kernel has none.

Only the KV-block axis (``b``) below is a Python loop. The running max
``m_i`` is a genuine sequential recurrence across ``b``, so it cannot be
batched away, but the head axis (``h``) and query-tile axis (``t``) have no
such dependency and are folded into a single batched matmul per block. For a
fixed block ``b``, the set of query tiles that still reach it is always a
suffix of the tile index (``n_blocks(t)`` is non-decreasing in ``t``), which is
what makes the per-block batching exact rather than approximate: it changes
which axis the Python loop runs over, not the quantity being counted. This
turns the wall-clock cost from ``O(n_heads * n_q_tiles * n_blocks)`` Python
iterations into ``O(n_blocks)`` batched GPU ops, verified to return bit-for-bit
identical ``(total, skipped[, partial])`` tuples to the original triple-loop
form across ten shapes (seqlens 511 to 4097, GQA ratios 1 to 16), with a 400x
to 650x wall-clock speedup at seqlen 4096 to 16384.
"""

from __future__ import annotations

import math

import torch

BLOCK_M = 128
BLOCK_N = 64


def _ceildiv(a: int, b: int) -> int:
    return -(-a // b)


def count_block_sparsity(
    q: torch.Tensor,
    k: torch.Tensor,
    threshold: float,
    softmax_scale: float | None = None,
    block_m: int = BLOCK_M,
    block_n: int = BLOCK_N,
    count_partial: bool = False,
) -> tuple[int, int] | tuple[int, int, int]:
    """Count total and fully-skipped K/V blocks for one causal sequence.

    Args:
        q: query tensor, ``[seqlen, n_heads, head_dim]``.
        k: key tensor, ``[seqlen, n_kv_heads, head_dim]``. GQA is expanded
            internally, so ``n_heads`` need not equal ``n_kv_heads``.
        threshold: the same probability ratio the kernel takes as
            ``skip_softmax_threshold``. Must be greater than 0; at 0 the kernel
            does not skip at all and the answer is trivially ``(total, 0)``.
        softmax_scale: defaults to ``1/sqrt(head_dim)``, as in the kernel.
        block_m: query tile size, must match the kernel's ``BLOCK_M``.
        block_n: key tile size, must match the kernel's ``BLOCK_N``.
        count_partial: also return the number of blocks where at least one row
            skipped. Those blocks still run their P@V matmul, so they do not
            count as sparsity and do not save time, but they *do* change the
            numerical result, because the skipped rows contribute nothing to
            the accumulator. Anything asking "should the output differ from
            dense?" needs this count, not the fully-skipped one.

    Returns:
        ``(total_blocks, skipped_blocks)``, or
        ``(total_blocks, skipped_blocks, partial_blocks)`` when
        ``count_partial``. Sparsity is ``skipped_blocks / total_blocks``.
    """
    if threshold <= 0.0:
        raise ValueError("threshold must be > 0; 0 disables skipping entirely")

    seqlen, n_heads, head_dim = q.shape
    n_kv_heads = k.shape[1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    log_threshold = math.log(threshold)

    if seqlen < block_n:
        return (0, 0, 0) if count_partial else (0, 0)

    device = q.device
    # The kernel reads bf16 and accumulates the dot product in fp32. Matching
    # the input dtype matters: the skip decision is a comparison of two nearby
    # scores, so rounding Q/K differently moves blocks across the boundary.
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    group = n_heads // n_kv_heads

    n_q_tiles = (seqlen + block_m - 1) // block_m
    pad_len = n_q_tiles * block_m - seqlen

    if pad_len:
        qf_padded = torch.nn.functional.pad(qf, (0, 0, 0, 0, 0, pad_len))
        row_idx = torch.arange(n_q_tiles * block_m, device=device)
        valid_row_mask = (row_idx < seqlen).reshape(n_q_tiles, block_m)
    else:
        qf_padded = qf
        valid_row_mask = None

    # [n_heads, n_q_tiles, block_m, head_dim], a view when pad_len == 0.
    q_tiles = qf_padded.permute(1, 0, 2).reshape(n_heads, n_q_tiles, block_m, head_dim)
    kv_head_idx = torch.arange(n_heads, device=device) // group
    k_by_head = kf.permute(1, 0, 2)  # [n_kv_heads, seqlen, head_dim], a view

    t_idx = torch.arange(n_q_tiles, device=device)
    main_end_t = (t_idx * block_m) // block_n  # main-loop blocks per tile
    max_blocks = int(main_end_t[-1].item()) + 2
    total = int(n_heads * (int(main_end_t.sum().item()) + 2 * n_q_tiles))
    q_starts = t_idx * block_m

    # Running max per (head, tile, row), raw (pre-scale) as in the kernel.
    m_i = torch.full(
        (n_heads, n_q_tiles, block_m), -float("inf"), device=device, dtype=torch.float32
    )

    skipped = 0
    partial = 0

    for b in range(max_blocks):
        kv_start = b * block_n
        if kv_start >= seqlen:
            continue
        kv_end = min(kv_start + block_n, seqlen)

        # Query tiles reached by block b are exactly a suffix [t_lo, n_q_tiles)
        # of the tile index, since n_blocks(t) = t*block_m//block_n + 2 is
        # non-decreasing in t.
        t_lo = _ceildiv((b - 1) * block_n, block_m)
        if t_lo >= n_q_tiles:
            continue
        num_valid_t = n_q_tiles - t_lo

        q_slice = q_tiles[:, t_lo:, :, :]  # [H, Tv, M, D]
        k_blk = k_by_head[kv_head_idx, kv_start:kv_end, :]  # [H, Nb, D]
        qk = torch.einsum("htmd,hnd->htmn", q_slice, k_blk) * softmax_scale

        # Applied unconditionally: for main-loop tiles (b < main_end(t)) every
        # key position is behind every query position in this tile, so the
        # mask is all-False and a no-op, matching the unmasked branch below.
        q_pos = q_starts[t_lo:].unsqueeze(1) + torch.arange(
            block_m, device=device
        ).unsqueeze(0)
        k_pos = torch.arange(kv_start, kv_end, device=device)
        causal_mask = k_pos.view(1, 1, 1, -1) > q_pos.view(1, num_valid_t, block_m, 1)
        qk = qk.masked_fill(causal_mask, -float("inf"))

        row_max = qk.max(dim=-1).values  # [H, Tv, M]

        m_i_slice = m_i[:, t_lo:, :]
        # A first block sees m_i = -inf, so row_max - m_i is +inf and never
        # skips; NaN from -inf minus -inf compares false too.
        skip = torch.nan_to_num(row_max - m_i_slice, nan=float("inf")) < log_threshold

        if valid_row_mask is not None:
            vrm = valid_row_mask[t_lo:, :]
            rows_count = vrm.sum(dim=-1)
            skip_counted = skip & vrm.unsqueeze(0)
        else:
            rows_count = torch.full(
                (num_valid_t,), block_m, device=device, dtype=torch.long
            )
            skip_counted = skip

        n_skipped_rows = skip_counted.sum(dim=-1)  # [H, Tv]
        fully_skipped = n_skipped_rows == rows_count.unsqueeze(0)
        skipped += int(fully_skipped.sum().item())
        if count_partial:
            partial += int(((n_skipped_rows > 0) & (~fully_skipped)).sum().item())

        # Skipped rows keep their old running max.
        m_i[:, t_lo:, :] = torch.where(
            skip, m_i_slice, torch.maximum(m_i_slice, row_max)
        )

    if count_partial:
        return total, skipped, partial
    return total, skipped
