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
"""

from __future__ import annotations

import math

import torch

BLOCK_M = 128
BLOCK_N = 64


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

    # The kernel reads bf16 and accumulates the dot product in fp32. Matching
    # the input dtype matters: the skip decision is a comparison of two nearby
    # scores, so rounding Q/K differently moves blocks across the boundary.
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    group = n_heads // n_kv_heads

    total = 0
    skipped = 0
    partial = 0
    n_q_tiles = (seqlen + block_m - 1) // block_m

    for h in range(n_heads):
        kh = kf[:, h // group, :]
        for t in range(n_q_tiles):
            q_start = t * block_m
            rows = min(block_m, seqlen - q_start)
            if rows <= 0:
                continue
            q_tile = qf[q_start : q_start + rows, h, :]

            # main loop blocks, then the two causal boundary blocks
            main_end = q_start // block_n
            n_blocks = main_end + 2

            # Running max per row, raw (pre-scale) as in the kernel.
            m_i = torch.full(
                (rows,), -float("inf"), device=q.device, dtype=torch.float32
            )

            for b in range(n_blocks):
                kv_start = b * block_n
                kv_end = min(kv_start + block_n, seqlen)
                if kv_start >= seqlen:
                    # Past the end: the kernel still issues the block with a
                    # mask, all scores are -inf, and it counts as visited.
                    total += 1
                    continue

                k_blk = kh[kv_start:kv_end, :]
                qk = (q_tile @ k_blk.T) * softmax_scale

                if b >= main_end:
                    # Causal boundary block: mask out keys after each query.
                    q_pos = torch.arange(
                        q_start, q_start + rows, device=q.device
                    ).unsqueeze(1)
                    k_pos = torch.arange(
                        kv_start, kv_end, device=q.device
                    ).unsqueeze(0)
                    qk = qk.masked_fill(k_pos > q_pos, -float("inf"))

                row_max = qk.max(dim=1).values
                skip = (row_max - m_i) < log_threshold
                # A first block sees m_i = -inf, so row_max - m_i is +inf and
                # never skips; NaN from -inf minus -inf compares false too.
                skip = torch.nan_to_num(
                    (row_max - m_i), nan=float("inf")
                ) < log_threshold

                total += 1
                n_skipped_rows = int(skip.sum())
                if n_skipped_rows == rows:
                    skipped += 1
                elif n_skipped_rows > 0:
                    partial += 1

                # Skipped rows keep their old running max.
                m_i = torch.where(skip, m_i, torch.maximum(m_i, row_max))

    if count_partial:
        return total, skipped, partial
    return total, skipped
