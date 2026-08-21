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

"""Sanity-check the reference sparsity counter against the shipped kernel.

The shipped kernel has no counters, so the strong check (comparing block counts
directly) needs an instrumented build and is not reproducible from this bundle.
That check was run during development and matched exactly on five shapes. What
this script does instead is test the reference's *predictions* against
observable kernel behaviour, which is weaker but needs nothing extra:

  [1] When the reference sees no row skipping anything anywhere, the kernel's
      output must be bit-identical to dense. If the reference under-counts, the
      kernel skipped something it did not know about and the outputs differ.
  [2] When the reference sees any row skip, the output must differ from dense.
      Note this is the *partial* count, not the sparsity count: a block where
      one row skips still runs its matmul, so it saves nothing, but it zeroes
      that row out of ``p`` and so changes the result. Testing sparsity here
      instead would wrongly demand bit-identical output across the whole band
      of thresholds where rows have started skipping but no block is yet
      skipped in full.
  [3] Sparsity must be monotonically non-decreasing in the threshold, in both
      the reference and, indirectly, the kernel.
  [4] ``defer_v_load=True`` must be bit-identical to ``False``. It changes when
      V is loaded, never which blocks are skipped.

Together these pin the threshold at which skipping begins, which is the part
most likely to be wrong (an off-by-one in log space, or comparing against the
post-block running max instead of the pre-block one).

Run inside the container:

    python scripts/verify_reference.py
"""

from __future__ import annotations

import sys

import torch

from sparsity_reference import count_block_sparsity
from tokenspeed_kernel_amd.ops.gfx950.attention.mha.prefill import (
    gluon_mha_prefill_gfx950,
)

HEAD_DIM = 128
DTYPE = torch.bfloat16

# (seqlen, n_q_heads, n_kv_heads)
SHAPES = [
    (512, 4, 1),
    (512, 8, 2),
    (1024, 4, 2),
    (2048, 8, 2),
    (2048, 4, 4),
]

# Spans the transition from "nothing skipped" to "most blocks skipped".
THRESHOLDS = [1e-9, 0.5, 0.9, 1.3, 2.0, 4.0]


def run(q, k, v, seqlen, threshold, defer):
    cu = torch.tensor([0, seqlen], device=q.device, dtype=torch.int32)
    return gluon_mha_prefill_gfx950(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu,
        cu_seqlens_cpu=[0, seqlen],
        max_seqlen=seqlen,
        skip_softmax_threshold=threshold,
        defer_v_load=defer,
    )


def main():
    failures = []
    print(f"{'shape':>16} {'threshold':>10} {'ref sparsity':>13} "
          f"{'partial':>8} {'differs':>8}  verdict")

    for seqlen, n_q, n_kv in SHAPES:
        torch.manual_seed(seqlen)
        q = torch.randn(seqlen, n_q, HEAD_DIM, device="cuda", dtype=DTYPE)
        k = torch.randn(seqlen, n_kv, HEAD_DIM, device="cuda", dtype=DTYPE)
        v = torch.randn(seqlen, n_kv, HEAD_DIM, device="cuda", dtype=DTYPE)

        dense = run(q, k, v, seqlen, 0.0, False)
        prev_skipped = -1

        for threshold in THRESHOLDS:
            total, skipped, partial = count_block_sparsity(
                q, k, threshold, count_partial=True
            )
            any_skip = (skipped + partial) > 0
            out = run(q, k, v, seqlen, threshold, False)
            differs = not torch.equal(out, dense)

            problems = []
            # [1] and [2]: the reference must agree with the kernel about
            # whether anything was skipped at all.
            if not any_skip and differs:
                problems.append("reference says nothing skipped, kernel differs")
            if any_skip and not differs:
                problems.append("reference says rows skipped, kernel identical")
            # [3] monotonicity in the threshold
            if skipped < prev_skipped:
                problems.append(f"sparsity fell from {prev_skipped} to {skipped}")
            prev_skipped = skipped
            # [4] defer_v_load must not change the result
            deferred = run(q, k, v, seqlen, threshold, True)
            if not torch.equal(deferred, out):
                problems.append("defer_v_load changed the output")

            pct = 100.0 * skipped / max(total, 1)
            verdict = "ok" if not problems else "FAIL: " + "; ".join(problems)
            if problems:
                failures.append((seqlen, n_q, n_kv, threshold, problems))
            print(f"{seqlen:>6} {n_q:>3}/{n_kv:<3} {threshold:>10} "
                  f"{skipped:>6}/{total:<6} {partial:>8} {str(differs):>8}  "
                  f"{verdict}")

        del q, k, v, dense
        torch.cuda.empty_cache()

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED.")
        return 1
    print("all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
