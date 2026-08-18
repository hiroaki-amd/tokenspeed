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

"""Registry-level smoke test for skip-softmax attention in ``mha_prefill``.

Covers the three dispatch outcomes for ``skip_softmax_threshold``:
a zero threshold leaves the pre-existing call form untouched; a non-zero
threshold on a platform with a kernel declaring the ``support_skip_softmax``
trait (currently only the gfx950 Gluon kernel) routes there; and a non-zero
threshold on a platform without such a kernel (e.g. gfx1250) or forced onto
a non-supporting solution raises ``NoKernelFoundError``.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel import mha_prefill
from tokenspeed_kernel.ops.attention import _attention_format_signature
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel

torch.manual_seed(7)

_DTYPE = torch.bfloat16
_NUM_Q_HEADS = 8
_NUM_KV_HEADS = 8
_HEAD_DIM = 128
_SEQLEN = 4096


def _qkv(seqlen: int, device: str):
    q = torch.randn((seqlen, _NUM_Q_HEADS, _HEAD_DIM), device=device, dtype=_DTYPE)
    k = torch.randn((seqlen, _NUM_KV_HEADS, _HEAD_DIM), device=device, dtype=_DTYPE)
    v = torch.randn((seqlen, _NUM_KV_HEADS, _HEAD_DIM), device=device, dtype=_DTYPE)
    return q, k, v


@pytest.mark.parametrize("solution", [None, "gluon", "triton"])
def test_mha_prefill_zero_threshold_preserves_legacy_call(
    device: str, require, solution: str | None
) -> None:
    """Adding ``skip_softmax_threshold`` must not disturb existing callers.

    Compares the pre-PR call form (threshold argument absent) against an
    explicit ``skip_softmax_threshold=0.0``; both must take the identical
    dense path. ``solution="triton"`` is the case that matters most:
    ``triton_mha_prefill`` has no ``skip_softmax_threshold`` parameter, so
    if the zero-threshold guard on the forwarded kwargs regressed into an
    unconditional forward, this raises ``TypeError``. Forcing ``"gluon"``
    alone cannot catch that, since that kernel accepts the argument.
    """
    require("attention", "mha_prefill", solution or "gluon", _DTYPE, "q")

    q, k, v = _qkv(_SEQLEN, device)
    common = {
        "q": q,
        "k": k,
        "v": v,
        "cu_seqlens": torch.tensor([0, _SEQLEN], device=device, dtype=torch.int32),
        "cu_seqlens_cpu": [0, _SEQLEN],
        "max_seqlen": _SEQLEN,
    }
    if solution is not None:
        common["solution"] = solution

    out_legacy = mha_prefill(**common)
    out_zero_threshold = mha_prefill(**common, skip_softmax_threshold=0.0)
    assert torch.equal(out_legacy, out_zero_threshold)


def test_mha_prefill_nonzero_threshold_routes_to_gfx950_gluon(
    mi350_platform, mi450_platform
) -> None:
    signature = _attention_format_signature(
        q=torch.empty((1, _NUM_Q_HEADS, _HEAD_DIM), dtype=_DTYPE),
        k=torch.empty((1, _NUM_KV_HEADS, _HEAD_DIM), dtype=_DTYPE),
        v=torch.empty((1, _NUM_KV_HEADS, _HEAD_DIM), dtype=_DTYPE),
    )
    traits = {"head_dim": _HEAD_DIM, "support_skip_softmax": True}

    selected = select_kernel(
        "attention",
        "mha_prefill",
        signature,
        traits=traits,
        platform=mi350_platform,
    )
    assert selected.name == "gluon_mha_prefill_gfx950"

    with pytest.raises(NoKernelFoundError):
        select_kernel(
            "attention",
            "mha_prefill",
            signature,
            traits=traits,
            platform=mi450_platform,
        )


def test_mha_prefill_skip_softmax_rejects_non_skip_softmax_solutions(
    device: str, require
) -> None:
    require("attention", "mha_prefill", "triton", _DTYPE, "q")

    cu_seqlens = torch.tensor([0, 256], device=device, dtype=torch.int32)
    cu_seqlens_cpu = [0, 256]
    q, k, v = _qkv(256, device)

    # Must fail in selection, not later: a bare `Exception` would also accept
    # the TypeError raised if the threshold were forwarded to a kernel that
    # does not accept it, hiding a broken trait gate.
    with pytest.raises(NoKernelFoundError):
        mha_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            max_seqlen=256,
            skip_softmax_threshold=1e-2,
            solution="triton",
        )
