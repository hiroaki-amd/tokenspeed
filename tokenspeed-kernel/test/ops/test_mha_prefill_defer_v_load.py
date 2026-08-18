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

"""Correctness test for ``defer_v_load`` in the Gluon MHA prefill kernel.

``defer_v_load=True`` selects the experimental deferred-V branch of
``process_attention_tile``'s main loop, which issues V's HBM->LDS load only
after the skip-softmax decision is known, so a fully-skipped KV block elides
V's memory traffic on top of its P@V matmul. It changes only when V is
loaded, never which blocks are skipped, so it must not change the output at
all.

This file re-runs the skip-softmax correctness checks from
``test_mha_prefill_skip_softmax.py`` with ``defer_v_load=True``, plus the
equivalence check that is the actual contract:
  [1] NO-REGRESSION  threshold=0.0 matches dense SDPA
  [2] TINY THRESHOLD threshold=1e-9 (skip enabled, ~never fires) matches
      dense SDPA
  [3] DEGRADATION    threshold>0 vs dense stays bounded
  [4] SKIP HAPPENS   larger threshold deviates from dense more than smaller
  [5] HIGH THRESHOLD threshold>1.0 (log2_threshold>0) produces finite output
  [6] EQUIVALENCE    bit-identical to ``defer_v_load=False`` at every
      threshold (the deferred load reorders memory traffic only; any
      difference means the deferred path skipped or accumulated a block the
      co-issue path did not), including with sinks, a sliding window, and
      ``return_lse``

[6] only constrains what the deferred branch does differently from the
co-issue branch; a defect in code both paths share shifts both results
identically and cancels. Checks [1]-[5] are what cover the shared code.
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
        "AMD CDNA4 is required for Gluon defer_v_load MHA prefill tests",
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


def _run(q, k, v, skip_softmax_threshold: float, defer_v_load: bool = True, **kwargs):
    cu_seqlens, cu_seqlens_cpu = _cu_seqlens()
    return gluon_mha_prefill_gfx950(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        max_seqlen=_SEQLEN,
        skip_softmax_threshold=skip_softmax_threshold,
        defer_v_load=defer_v_load,
        **kwargs,
    )


def test_defer_v_load_no_regression() -> None:
    q, k, v = _qkv()
    dense = _dense_ref(q, k, v)
    out0 = _run(q, k, v, skip_softmax_threshold=0.0)
    assert torch.isfinite(out0).all()
    assert _rel_err(out0, dense) < _NO_REGRESSION_TOL


def test_defer_v_load_tiny_threshold_matches_dense() -> None:
    q, k, v = _qkv()
    dense = _dense_ref(q, k, v)
    out = _run(q, k, v, skip_softmax_threshold=_TINY_THRESHOLD)
    assert torch.isfinite(out).all()
    assert _rel_err(out, dense) < _NO_REGRESSION_TOL


@pytest.mark.parametrize("threshold", _THRESHOLDS)
def test_defer_v_load_degrades_gracefully(threshold: float) -> None:
    q, k, v = _qkv()
    dense = _dense_ref(q, k, v)
    out = _run(q, k, v, skip_softmax_threshold=threshold)
    assert torch.isfinite(out).all()
    assert _rel_err(out, dense) < _DEGRADATION_BOUND


def test_defer_v_load_actually_skips() -> None:
    q, k, v = _qkv()
    dense = _dense_ref(q, k, v)
    r_lo = _rel_err(_run(q, k, v, skip_softmax_threshold=_THRESHOLDS[0]), dense)
    r_hi = _rel_err(_run(q, k, v, skip_softmax_threshold=_THRESHOLDS[-1]), dense)
    assert r_hi > r_lo


@pytest.mark.parametrize("threshold", [2.0, 5.0, 12.0])
def test_defer_v_load_high_threshold_no_nan(threshold: float) -> None:
    """skip_softmax_threshold > 1.0 (log2_threshold > 0) must stay finite

    with defer_v_load=True, mirroring the co-issue main loop's regression
    guard in ``test_mha_prefill_skip_softmax.py``.
    """
    q, k, v = _qkv()
    out = _run(q, k, v, skip_softmax_threshold=threshold)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "threshold", [0.0, _TINY_THRESHOLD, *_THRESHOLDS, 2.0, 5.0, 12.0]
)
def test_defer_v_load_matches_co_issue_exactly(threshold: float) -> None:
    """Deferring V's load must not change the result at all.

    Both paths run the same skip-softmax decision on the same K blocks and
    accumulate the same P@V products; only the point at which V's HBM->LDS
    copy is issued differs. Requiring bit-identical output catches defects
    confined to the deferred branch itself (where issue_load_v sits relative
    to the skip decision, its wait_group depth) that the dense-SDPA checks
    above would wave through, since their tolerances are far looser than
    such a defect's effect.

    This comparison is blind to anything that hits both branches equally:
    v_offsets advances outside the DEFER_V_LOAD conditional, so breaking
    that advance shifts the deferred and co-issue results identically and
    cancels in the difference. Defects in code shared by both paths are the
    dense-SDPA checks' job, not this one's.
    """
    q, k, v = _qkv()
    deferred = _run(q, k, v, skip_softmax_threshold=threshold, defer_v_load=True)
    co_issue = _run(q, k, v, skip_softmax_threshold=threshold, defer_v_load=False)
    assert torch.equal(deferred, co_issue)


@pytest.mark.parametrize("threshold", [0.0, 3e-1, 12.0])
@pytest.mark.parametrize("feature", ["sinks", "sliding"])
def test_defer_v_load_matches_co_issue_with_other_features(
    threshold: float, feature: str
) -> None:
    """The bit-identical contract must hold for the other feature paths too.

    ``sinks`` changes the initial running max, and ``sliding`` routes to the
    sliding kernel, where the launcher forces the deferred path off; both
    must still land on exactly the co-issue result.
    """
    q, k, v = _qkv()
    if feature == "sinks":
        kwargs = {
            "sinks": torch.randn((_NUM_Q_HEADS,), device="cuda", dtype=torch.float32)
        }
    else:
        kwargs = {"window_left": 256}
    deferred = _run(q, k, v, threshold, defer_v_load=True, **kwargs)
    co_issue = _run(q, k, v, threshold, defer_v_load=False, **kwargs)
    assert torch.equal(deferred, co_issue)


@pytest.mark.parametrize("threshold", [0.0, 3e-1, 12.0])
def test_defer_v_load_matches_co_issue_with_lse(threshold: float) -> None:
    """Deferring V's load must not perturb the returned LSE either."""
    q, k, v = _qkv()
    out_d, lse_d = _run(q, k, v, threshold, defer_v_load=True, return_lse=True)
    out_c, lse_c = _run(q, k, v, threshold, defer_v_load=False, return_lse=True)
    assert torch.equal(out_d, out_c)
    assert torch.equal(lse_d, lse_c)
