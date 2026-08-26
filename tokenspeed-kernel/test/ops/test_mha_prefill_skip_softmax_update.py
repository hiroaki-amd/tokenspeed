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

"""The skip decision is per row, but only a unanimous block changes anything.

BLASST tests each query row of a tile against the running max, then takes a
block-level decision from those per-row votes. The kernel elides a K/V block
only when every row of the tile votes to skip it: the exp2 over
BLOCK_M x BLOCK_N, the row sum, the rescale of the BLOCK_M x HEAD_DIM
accumulator, the P@V matmul and, under ``defer_v_load``, V's HBM load. On any
block where at least one row dissents, the individual votes are discarded and
every row takes the ordinary online-softmax update.

That last part is the contract this file pins, and it is the one that is easy
to get wrong. Dropping a dissenting block's contribution for the rows that did
vote to skip is the obvious reading of the paper's Algorithm 1, and it is both
less accurate and slower: holding a row's running max back makes that row less
likely to clear the threshold on later blocks, so the block-level skip rate
falls as well. The BLASST authors' own Hopper kernel does not do it either
(TensorRT-LLM ``cpp/kernels/fmha_v2/src/fmha/warpspec/epilogue.h``: when the
warpgroup vote fails, the exp loop runs over every row and the per-row bits are
dropped).

Checks (bf16, causal, fixed seed):
  [1] VOTES ARE INERT   at a threshold high enough that a large fraction of
      rows vote to skip but no block is unanimous, the output is bit-identical
      to a threshold too small for any row to vote at all. Covered across GQA
      ratios, both ``defer_v_load`` settings, sinks, ``return_lse``, a sliding
      window and a ragged batch, since the vote sits in code every one of those
      paths runs. The row rates at these thresholds are measured in
      ``_ROW_VOTE_RATES`` below, so this is not a vacuous assertion.
  [2] ELISION FIRES     past that range blocks do go unanimous, the output does
      move, and it stays finite. Without this, [1] would also pass on a kernel
      that never skipped anything.
  [3] NO REGRESSION     with skipping off, the result still matches dense SDPA.

[1] and [2] constrain the block-level rule only. How far the elided output may
drift from dense is covered by ``test_mha_prefill_skip_softmax.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel_amd.ops.gfx950.attention.mha.prefill import (
    gluon_mha_prefill_gfx950,
)
from utils import is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for Gluon skip-softmax MHA prefill tests",
        allow_module_level=True,
    )


_SEQLEN = 4096
_HEAD_DIM = 128
_DTYPE = torch.bfloat16
_NO_REGRESSION_TOL = 5e-3
# Small enough that no row clears the skip test at these score distributions,
# so nothing votes and nothing is elided. This is the baseline [1] compares
# against: it compiles the same ENABLE_SKIP_SOFTMAX=True kernel, so any
# difference from it is the skipping and not a build difference.
_TINY_THRESHOLD = 1e-9
# Thresholds where rows vote in quantity but no block goes unanimous, measured
# on the shapes below at seed 0. Per-row vote rate against unanimous-block rate
# at 8 Q-heads / 2 KV-heads, seqlen 4096:
#
#   0.001    0.00% rows   0.000% blocks
#   0.01     0.00% rows   0.000% blocks
#   0.1      1.03% rows   0.000% blocks
#   0.3     31.90% rows   0.000% blocks
#
# 0.3 is the load-bearing one: nearly a third of row-block pairs vote to skip
# there and the output must still not move by a bit.
_ROW_VOTE_RATES = [1e-3, 1e-2, 1e-1, 0.3]
# Past the vote-only range: blocks do go unanimous here (1.16% at 0.9 on the
# shape above), so the output is expected to move.
_ELIDING_THRESHOLDS = [0.7, 0.9]
_GQA_SHAPES = [(8, 8), (8, 2)]


def _qkv(n_heads: int, n_kv_heads: int, total_tokens: int, seed: int = 0):
    torch.manual_seed(seed)
    q = torch.randn((total_tokens, n_heads, _HEAD_DIM), device="cuda", dtype=_DTYPE)
    k = torch.randn((total_tokens, n_kv_heads, _HEAD_DIM), device="cuda", dtype=_DTYPE)
    v = torch.randn((total_tokens, n_kv_heads, _HEAD_DIM), device="cuda", dtype=_DTYPE)
    return q, k, v


def _dense_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Per-sequence causal dense reference in fp32. [S,H,D] layout, one sequence."""
    n_heads, n_kv_heads = q.shape[1], k.shape[1]
    qt, kt, vt = (x.transpose(0, 1).float().unsqueeze(0) for x in (q, k, v))
    if n_kv_heads != n_heads:
        repeat = n_heads // n_kv_heads
        kt = kt.repeat_interleave(repeat, dim=1)
        vt = vt.repeat_interleave(repeat, dim=1)
    out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
    return out.squeeze(0).transpose(0, 1).to(q.dtype)


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (
        (a.float() - b.float()).abs().mean() / b.float().abs().mean().clamp_min(1e-6)
    ).item()


def _run(q, k, v, seqlens: list[int], threshold: float, **kwargs):
    cu = [0]
    for s in seqlens:
        cu.append(cu[-1] + s)
    return gluon_mha_prefill_gfx950(
        q=q,
        k=k,
        v=v,
        cu_seqlens=torch.tensor(cu, device="cuda", dtype=torch.int32),
        cu_seqlens_cpu=cu,
        max_seqlen=max(seqlens),
        skip_softmax_threshold=threshold,
        **kwargs,
    )


@pytest.mark.parametrize("n_heads,n_kv_heads", _GQA_SHAPES)
@pytest.mark.parametrize("threshold", _ROW_VOTE_RATES)
@pytest.mark.parametrize("defer_v_load", [False, True])
def test_row_votes_alone_change_nothing(
    n_heads: int, n_kv_heads: int, threshold: float, defer_v_load: bool
) -> None:
    """[1] Rows vote, but a block that is not unanimous must be exact."""
    q, k, v = _qkv(n_heads, n_kv_heads, _SEQLEN)
    kwargs = {"defer_v_load": defer_v_load}
    voting = _run(q, k, v, [_SEQLEN], threshold, **kwargs)
    inert = _run(q, k, v, [_SEQLEN], _TINY_THRESHOLD, **kwargs)
    assert torch.isfinite(voting).all()
    assert torch.equal(voting, inert)


@pytest.mark.parametrize("threshold", _ROW_VOTE_RATES)
def test_row_votes_alone_preserve_lse(threshold: float) -> None:
    """[1] l_i and m_i are what a mishandled vote would corrupt first."""
    q, k, v = _qkv(8, 2, _SEQLEN)
    o_voting, lse_voting = _run(q, k, v, [_SEQLEN], threshold, return_lse=True)
    o_inert, lse_inert = _run(q, k, v, [_SEQLEN], _TINY_THRESHOLD, return_lse=True)
    assert torch.isfinite(lse_voting).all()
    assert torch.equal(o_voting, o_inert)
    assert torch.equal(lse_voting, lse_inert)


@pytest.mark.parametrize("threshold", _ROW_VOTE_RATES)
def test_row_votes_alone_with_sinks(threshold: float) -> None:
    """[1] Sinks change the m_i initialization, which the vote reads."""
    q, k, v = _qkv(8, 2, _SEQLEN)
    sinks = torch.randn((8,), device="cuda", dtype=torch.float32)
    voting = _run(q, k, v, [_SEQLEN], threshold, sinks=sinks)
    inert = _run(q, k, v, [_SEQLEN], _TINY_THRESHOLD, sinks=sinks)
    assert torch.isfinite(voting).all()
    assert torch.equal(voting, inert)


@pytest.mark.parametrize("threshold", _ROW_VOTE_RATES)
def test_row_votes_alone_sliding_window(threshold: float) -> None:
    """[1] The sliding kernel is the HAS_INVALID variant and votes separately.

    It also reaches the unanimous case for a reason unrelated to the
    threshold: a tile the window has moved past is masked out entirely, so
    every row's max is -inf and every row votes. Eliding is right there, since
    such a tile contributes exactly zero, and this pins that it stays exact.
    """
    q, k, v = _qkv(8, 8, _SEQLEN)
    voting = _run(q, k, v, [_SEQLEN], threshold, window_left=256)
    inert = _run(q, k, v, [_SEQLEN], _TINY_THRESHOLD, window_left=256)
    assert torch.isfinite(voting).all()
    assert torch.equal(voting, inert)


@pytest.mark.parametrize("threshold", _ROW_VOTE_RATES)
def test_row_votes_alone_ragged_batch(threshold: float) -> None:
    """[1] Includes a sequence shorter than BLOCK_M, which takes its own path."""
    seqlens = [1024, 64, 2048, 512]
    q, k, v = _qkv(8, 2, sum(seqlens), seed=1)
    voting = _run(q, k, v, seqlens, threshold)
    inert = _run(q, k, v, seqlens, _TINY_THRESHOLD)
    assert torch.isfinite(voting).all()
    assert torch.equal(voting, inert)


@pytest.mark.parametrize("threshold", _ELIDING_THRESHOLDS)
def test_elision_actually_fires(threshold: float) -> None:
    """[2] Blocks do go unanimous past the vote-only range, and the output moves.

    Guards against the tests above passing on a kernel that skips nothing at
    all: without this, "bit-identical" would be trivially satisfied.
    """
    q, k, v = _qkv(8, 2, _SEQLEN)
    elided = _run(q, k, v, [_SEQLEN], threshold)
    inert = _run(q, k, v, [_SEQLEN], _TINY_THRESHOLD)
    assert torch.isfinite(elided).all()
    assert not torch.equal(elided, inert)


@pytest.mark.parametrize("threshold", [0.0, _TINY_THRESHOLD])
def test_inert_without_skipping(threshold: float) -> None:
    """[3] With nothing to vote or elide, the result still matches dense."""
    q, k, v = _qkv(8, 2, _SEQLEN)
    out = _run(q, k, v, [_SEQLEN], threshold)
    assert torch.isfinite(out).all()
    assert _rel_err(out, _dense_ref(q, k, v)) < _NO_REGRESSION_TOL
