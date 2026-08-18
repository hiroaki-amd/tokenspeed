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

"""Correctness test for skip-softmax attention in the Gluon MHA prefill kernel.

Skips a K/V block's P@V matmul for query rows whose per-block max score falls
below the running softmax max by more than a threshold. This is a
compute-skip-only implementation: the V-block load itself is not skipped, so
no memory-bandwidth savings are expected here, only matmul-skip savings.

Exposed via ``gluon_mha_prefill_gfx950(..., skip_softmax_threshold=X)``
(``0.0`` disables skipping, exact dense attention).

Checks (bf16, causal):
  [1] NO-REGRESSION  threshold=0.0 matches dense SDPA
  [2] TINY THRESHOLD threshold=1e-9 (skip enabled, ~never fires) matches
      dense SDPA (threshold=0.0 compiles a separate
      ENABLE_SKIP_SOFTMAX=False kernel variant with the skip branch elided
      entirely, so it never exercises the skip-enabled branch's
      masking/accumulation arithmetic; this check does, while sparsity
      stays close to 0%)
  [3] DEGRADATION    threshold>0 vs dense stays bounded
  [4] SKIP HAPPENS   larger threshold deviates from dense more than smaller
  [5] HIGH THRESHOLD threshold>1.0 (log2_threshold>0) produces finite output
      (regression guard for a pre/post-block running-max comparison bug: the
      skip decision must use the PRE-block max, not the POST-block max, or
      the first KV block of every tile is forced to skip unconditionally
      whenever threshold>1.0, producing all-NaN output)
  [6] MASKING PATHS  sliding window (the HAS_INVALID kernel variant) and
      attention sinks (where m_i starts at the sink value instead of -inf,
      so an early block can legitimately be skipped) both stay finite
      across the threshold range
  [7] LSE            the returned LSE stays finite, is never above the dense
      LSE, and degrades monotonically with the output
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
_NUM_Q_HEADS = 8
_NUM_KV_HEADS = 8
_HEAD_DIM = 128
_DTYPE = torch.bfloat16
_NO_REGRESSION_TOL = 5e-3
_DEGRADATION_BOUND = 0.6
_THRESHOLDS = [1e-3, 1e-2, 5e-2, 1e-1, 3e-1]
_TINY_THRESHOLD = 1e-9


def _qkv(seed: int = 0):
    torch.manual_seed(seed)
    shape = (_SEQLEN, _NUM_Q_HEADS, _HEAD_DIM)
    kv_shape = (_SEQLEN, _NUM_KV_HEADS, _HEAD_DIM)
    q = torch.randn(shape, device="cuda", dtype=_DTYPE)
    k = torch.randn(kv_shape, device="cuda", dtype=_DTYPE)
    v = torch.randn(kv_shape, device="cuda", dtype=_DTYPE)
    return q, k, v


def _dense_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Single-sequence causal dense reference; fp32 SDPA math. [S,H,D] layout."""
    qt, kt, vt = (x.transpose(0, 1).float().unsqueeze(0) for x in (q, k, v))
    out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
    return out.squeeze(0).transpose(0, 1).to(q.dtype)


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (
        (a.float() - b.float()).abs().mean() / b.float().abs().mean().clamp_min(1e-6)
    ).item()


def _cu_seqlens():
    cu = torch.tensor([0, _SEQLEN], device="cuda", dtype=torch.int32)
    return cu, [0, _SEQLEN]


def _run(q, k, v, skip_softmax_threshold: float, **kwargs):
    cu_seqlens, cu_seqlens_cpu = _cu_seqlens()
    return gluon_mha_prefill_gfx950(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        max_seqlen=_SEQLEN,
        skip_softmax_threshold=skip_softmax_threshold,
        **kwargs,
    )


def test_skip_softmax_no_regression() -> None:
    q, k, v = _qkv()
    dense = _dense_ref(q, k, v)
    out0 = _run(q, k, v, skip_softmax_threshold=0.0)
    assert torch.isfinite(out0).all()
    assert _rel_err(out0, dense) < _NO_REGRESSION_TOL


def test_skip_softmax_tiny_threshold_matches_dense() -> None:
    """threshold=0.0 compiles a separate ENABLE_SKIP_SOFTMAX=False kernel

    variant with the skip branch elided entirely, so
    ``test_skip_softmax_no_regression`` never exercises the skip-enabled
    branch's masking/accumulation arithmetic. A tiny nonzero threshold
    compiles the ENABLE_SKIP_SOFTMAX=True variant while keeping sparsity
    close to 0%, so output should still match dense within the same
    tolerance.
    """
    q, k, v = _qkv()
    dense = _dense_ref(q, k, v)
    out = _run(q, k, v, skip_softmax_threshold=_TINY_THRESHOLD)
    assert torch.isfinite(out).all()
    assert _rel_err(out, dense) < _NO_REGRESSION_TOL


@pytest.mark.parametrize("threshold", _THRESHOLDS)
def test_skip_softmax_degrades_gracefully(threshold: float) -> None:
    q, k, v = _qkv()
    dense = _dense_ref(q, k, v)
    out = _run(q, k, v, skip_softmax_threshold=threshold)
    assert torch.isfinite(out).all()
    assert _rel_err(out, dense) < _DEGRADATION_BOUND


def test_skip_softmax_actually_skips() -> None:
    q, k, v = _qkv()
    dense = _dense_ref(q, k, v)
    r_lo = _rel_err(_run(q, k, v, skip_softmax_threshold=_THRESHOLDS[0]), dense)
    r_hi = _rel_err(_run(q, k, v, skip_softmax_threshold=_THRESHOLDS[-1]), dense)
    assert r_hi > r_lo


@pytest.mark.parametrize("threshold", [2.0, 5.0, 12.0])
def test_skip_softmax_high_threshold_no_nan(threshold: float) -> None:
    """skip_softmax_threshold > 1.0 (log2_threshold > 0) must stay finite.

    Regresses a bug where comparing the skip decision against the
    post-block running max (instead of the pre-block max) forced the first
    KV block of every tile to skip unconditionally once threshold > 1.0,
    producing all-NaN output.
    """
    q, k, v = _qkv()
    out = _run(q, k, v, skip_softmax_threshold=threshold)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("window_left", [1, 64, 256])
@pytest.mark.parametrize("threshold", [0.0, _TINY_THRESHOLD, 3e-1, 12.0])
def test_skip_softmax_sliding_window_stays_finite(
    threshold: float, window_left: int
) -> None:
    """Skip-softmax must not break the sliding-window masking path.

    With window_left >= 0 and no sinks the kernel compiles its HAS_INVALID
    variant, which guards the online-softmax state against rows whose
    running max is still -inf. The skip decision is inserted upstream of
    that guard and feeds it m_new, so this pins that the two compose
    across the threshold range and several window sizes rather than only
    at the default full-attention configuration.
    """
    q, k, v = _qkv()
    out = _run(q, k, v, skip_softmax_threshold=threshold, window_left=window_left)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("threshold", [0.0, _TINY_THRESHOLD, 3e-1, 12.0])
def test_skip_softmax_with_sinks_stays_finite(threshold: float) -> None:
    """Attention sinks initialise m_i to the sink value rather than -inf.

    That changes what the skip comparison sees on a tile's first KV block
    (a finite baseline instead of +inf), so it is the one configuration
    where an early block can legitimately be skipped.
    """
    q, k, v = _qkv()
    sinks = torch.randn((_NUM_Q_HEADS,), device="cuda", dtype=torch.float32)
    out = _run(q, k, v, skip_softmax_threshold=threshold, sinks=sinks)
    assert torch.isfinite(out).all()


def test_skip_softmax_lse_is_finite_and_degrades_with_output() -> None:
    """The returned LSE is approximate in the same way the output is.

    Skipped blocks contribute nothing to the softmax denominator, so the
    LSE is biased low rather than merely noisy. This pins that it stays
    finite and tracks the output's degradation, so callers combining
    partial results through the LSE (chunked prefill, speculative
    decoding) have a documented expectation rather than an accident.
    """
    q, k, v = _qkv()
    out_dense, lse_dense = _run(q, k, v, skip_softmax_threshold=0.0, return_lse=True)
    assert torch.isfinite(lse_dense).all()

    prev_lse_err = -1.0
    prev_out_err = -1.0
    for threshold in (_TINY_THRESHOLD, 3e-1, 12.0):
        out, lse = _run(q, k, v, skip_softmax_threshold=threshold, return_lse=True)
        assert torch.isfinite(out).all()
        assert torch.isfinite(lse).all()
        # Skipping only ever drops terms from the denominator.
        assert (lse <= lse_dense + _NO_REGRESSION_TOL).all()
        lse_err = _rel_err(lse, lse_dense)
        out_err = _rel_err(out, out_dense)
        assert lse_err >= prev_lse_err
        assert out_err >= prev_out_err
        prev_lse_err, prev_out_err = lse_err, out_err
