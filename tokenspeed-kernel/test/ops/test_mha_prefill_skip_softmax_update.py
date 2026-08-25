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

"""Correctness test for ``skip_softmax_update`` in the Gluon MHA prefill kernel.

When every row of a tile votes to skip a KV block, the online-softmax update
for that block computes the identity: ``m_new`` is already ``m_i`` on every
row, so ``alpha = exp2(0) = 1``, and ``p`` is zeroed on every row, so
``l_ij = 0``, leaving ``l_i * 1 + 0`` and ``acc * 1``. ``skip_softmax_update``
elides the update outright in that case, which is what removes the exp2 over
BLOCK_M x BLOCK_N, the row sum, and the rescale of the BLOCK_M x HEAD_DIM
accumulator. Without it the kernel skips only the consumer of the softmax
(the P@V matmul and, under ``defer_v_load``, V's load) while still evaluating
the softmax itself.

The bf16 output is bit-identical either way, and that is the contract this
file pins. The fp32 LSE is not, but not for the reason the elision might
suggest. ``skip_softmax_update`` is a constexpr, so the two settings compile
to two kernels, and the branch shifts instruction selection around the row
sum: one build emits 57 ``v_pk_add_f32``, the other 58. That reassociates the
reduction tree, which moves ``l_i`` by a few ULP on the rows it touches. The
divergence is therefore a property of the two builds, not of skipping: it is
present at a threshold small enough that no block can clear the skip test,
identical in position and magnitude from 1e-9 up to 1.0, and both builds sit
the same distance from an fp64 dense reference. Measured worst case over the
shapes below is 37 ULP, 2.5e-6 relative. The output survives because bf16
rounding absorbs it.

Under a sliding window the elision genuinely is not safe, which is a separate
matter: a tile the window has moved past is masked out entirely, ``row_max``
is -inf against a finite ``m_i``, the difference is -inf, and every row votes
to skip against a running max that is not theirs. The launcher clears the flag
there, and [3] pins that down rather than leaving it to be rediscovered.

  [1] EQUIVALENCE  output is bit-identical to ``skip_softmax_update=False`` at
      every threshold, which is what makes the elision safe. Covered across
      GQA ratios, both ``defer_v_load`` settings, sinks, ``return_lse`` and a
      ragged batch, since the elision sits in code every one of those paths
      runs.
  [2] LSE          the returned LSE agrees to the reassociation tolerance, and
      the same tolerance holds at a threshold too small to skip anything. The
      second half is the load-bearing one: it shows the gap is the build
      difference described above and not error the elision introduces.
  [3] SLIDING      the flag is forced off for the sliding-window kernel, so
      requesting it must not change that kernel's output.
  [4] NO-SKIP      with skipping off, or at a threshold too small to fire, the
      flag is inert and the result still matches dense SDPA.

[1] only constrains the elision against performing the update; a defect in
the shared skip decision moves both arms together and cancels. The dense
comparisons in ``test_mha_prefill_skip_softmax.py`` are what cover that.
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
# The two constexpr settings compile to two kernels whose row-sum reduction
# trees are associated differently, so the fp32 LSE they return differs by a
# few ULP. Measured worst case over the shapes here is 2.5e-6 relative; this
# is an order above that and still far below anything the LSE feeds.
_LSE_REASSOC_TOL = 2e-5
# Small enough that no block clears the skip test at these score
# distributions, so the flag has nothing to elide.
_TINY_THRESHOLD = 1e-9
# 0.3 is past 1.0 in log2 space for some rows, so it drives sparsity high
# enough that all_skip fires on a large fraction of blocks.
_THRESHOLDS = [1e-3, 1e-2, 1e-1, 0.3]
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
@pytest.mark.parametrize("threshold", _THRESHOLDS)
@pytest.mark.parametrize("defer_v_load", [False, True])
def test_update_elision_is_bit_identical(
    n_heads: int, n_kv_heads: int, threshold: float, defer_v_load: bool
) -> None:
    """[1] Eliding an identity update must not change a single bit."""
    q, k, v = _qkv(n_heads, n_kv_heads, _SEQLEN)
    kwargs = {"defer_v_load": defer_v_load}
    elided = _run(q, k, v, [_SEQLEN], threshold, skip_softmax_update=True, **kwargs)
    computed = _run(q, k, v, [_SEQLEN], threshold, skip_softmax_update=False, **kwargs)
    assert torch.isfinite(elided).all()
    assert torch.equal(elided, computed)


@pytest.mark.parametrize("threshold", [_TINY_THRESHOLD] + _THRESHOLDS)
def test_update_elision_preserves_lse(threshold: float) -> None:
    """[2] l_i is what the elision touches, so check it directly.

    ``_TINY_THRESHOLD`` is in the parameter list on purpose. No block can
    clear the skip test there, so nothing is elided, and the LSE gap that
    remains is the reduction reassociation between the two builds rather than
    anything the elision did. Whatever tolerance this case needs is the
    tolerance the skipping cases are entitled to.
    """
    q, k, v = _qkv(8, 2, _SEQLEN)
    o_elided, lse_elided = _run(
        q, k, v, [_SEQLEN], threshold, skip_softmax_update=True, return_lse=True
    )
    o_computed, lse_computed = _run(
        q, k, v, [_SEQLEN], threshold, skip_softmax_update=False, return_lse=True
    )
    assert torch.isfinite(lse_elided).all()
    assert torch.equal(o_elided, o_computed)
    rel = (lse_elided - lse_computed).abs() / lse_computed.abs().clamp_min(1e-6)
    assert rel.max().item() < _LSE_REASSOC_TOL


@pytest.mark.parametrize("threshold", _THRESHOLDS)
def test_update_elision_with_sinks(threshold: float) -> None:
    """[1] Sinks change the m_i initialization, which the skip test reads."""
    q, k, v = _qkv(8, 2, _SEQLEN)
    sinks = torch.randn((8,), device="cuda", dtype=torch.float32)
    elided = _run(q, k, v, [_SEQLEN], threshold, sinks=sinks, skip_softmax_update=True)
    computed = _run(
        q, k, v, [_SEQLEN], threshold, sinks=sinks, skip_softmax_update=False
    )
    assert torch.isfinite(elided).all()
    assert torch.equal(elided, computed)


@pytest.mark.parametrize("threshold", _THRESHOLDS)
def test_update_elision_ragged_batch(threshold: float) -> None:
    """[1] Includes a sequence shorter than BLOCK_M, which takes its own path."""
    seqlens = [1024, 64, 2048, 512]
    q, k, v = _qkv(8, 2, sum(seqlens))
    elided = _run(q, k, v, seqlens, threshold, skip_softmax_update=True)
    computed = _run(q, k, v, seqlens, threshold, skip_softmax_update=False)
    assert torch.isfinite(elided).all()
    assert torch.equal(elided, computed)


@pytest.mark.parametrize("threshold", _THRESHOLDS)
def test_sliding_window_ignores_the_flag(threshold: float) -> None:
    """[3] The launcher forces the flag off here; asking for it changes nothing.

    Not a free assertion: eliding the update on the sliding kernel does move
    results, because a tile the window has moved past is masked out entirely
    and every row then votes to skip against a finite running max. The flag
    is cleared for that reason, and this pins the clearing down.
    """
    q, k, v = _qkv(8, 8, _SEQLEN)
    on = _run(q, k, v, [_SEQLEN], threshold, window_left=256, skip_softmax_update=True)
    off = _run(
        q, k, v, [_SEQLEN], threshold, window_left=256, skip_softmax_update=False
    )
    assert torch.isfinite(on).all()
    assert torch.equal(on, off)


@pytest.mark.parametrize("threshold", [0.0, _TINY_THRESHOLD])
@pytest.mark.parametrize("skip_softmax_update", [False, True])
def test_inert_without_skipping(threshold: float, skip_softmax_update: bool) -> None:
    """[4] With nothing to elide, the flag must not perturb a dense result."""
    q, k, v = _qkv(8, 2, _SEQLEN)
    out = _run(q, k, v, [_SEQLEN], threshold, skip_softmax_update=skip_softmax_update)
    assert torch.isfinite(out).all()
    assert _rel_err(out, _dense_ref(q, k, v)) < _NO_REGRESSION_TOL
