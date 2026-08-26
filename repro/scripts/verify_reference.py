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

  [1] When the reference sees no block skipped in full, the kernel's output
      must be bit-identical to the baseline. A block whose vote was not
      unanimous has its votes discarded and runs exactly as it would with
      skipping off, so votes alone must not perturb the result. This is the
      check that would catch the reference applying votes per row: under that
      rule a partially voted block *does* change the output, and the two cannot
      both be true.
  [2] When the reference sees any block skipped in full, the output must
      differ from the baseline. Together with [1] this pins sparsity to the
      exact threshold at which the kernel's output starts moving.
  [3] Sparsity must be monotonically non-decreasing in the threshold, in both
      the reference and, indirectly, the kernel.
  [4] ``defer_v_load=True`` must be bit-identical to ``False``. It changes when
      V is loaded, never which blocks are skipped.

Together these pin the threshold at which skipping begins, which is the part
most likely to be wrong (an off-by-one in log space, or comparing against the
post-block running max instead of the pre-block one).

The partial count is still gathered and printed, but it is diagnostic only: it
shows how wide the band is where rows vote in quantity and nothing is yet
elided. It is deliberately *not* asserted against the output.

The baseline for [1], [2] and [4] is ``skip_softmax_threshold=1e-9``, not
``0.0``. See the comment at the ``dense =`` line: 0.0 is a different
compilation and differs in the last bits for reasons that have nothing to do
with skipping.

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

# Spans the transition from "nothing skipped" to "most blocks skipped". Every
# one of these is measured against the 1e-9 baseline below, so the band where
# rows vote but no block is unanimous has to be inside this list for [1] to be
# testing anything: at 0.5 and 0.9 the small shapes have hundreds of votes and
# zero skipped blocks, and must still come out bit-identical.
THRESHOLDS = [0.5, 0.9, 1.3, 2.0, 4.0]


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
    print(
        f"{'shape':>16} {'threshold':>10} {'ref sparsity':>13} "
        f"{'partial':>8} {'differs':>8}  verdict"
    )

    for seqlen, n_q, n_kv in SHAPES:
        torch.manual_seed(seqlen)
        q = torch.randn(seqlen, n_q, HEAD_DIM, device="cuda", dtype=DTYPE)
        k = torch.randn(seqlen, n_kv, HEAD_DIM, device="cuda", dtype=DTYPE)
        v = torch.randn(seqlen, n_kv, HEAD_DIM, device="cuda", dtype=DTYPE)

        # The baseline is threshold=1e-9, not 0.0. At 0.0 the skip branch is
        # compiled out and the kernel takes the static scheduler, which
        # associates the row-sum reduction differently, so it disagrees with
        # every skipping build in the last bit or two even when nothing is
        # skipped. That is a legitimate floating-point difference, not a skip,
        # but comparing against it makes bit-identity untestable. 1e-9 is the
        # same compilation as the thresholds under test, with a threshold too
        # small for any row to vote, which is the baseline these checks want.
        dense = run(q, k, v, seqlen, 1e-9, False)
        prev_skipped = -1

        for threshold in THRESHOLDS:
            total, skipped, partial = count_block_sparsity(
                q, k, threshold, count_partial=True
            )
            any_skip = skipped > 0
            out = run(q, k, v, seqlen, threshold, False)
            differs = not torch.equal(out, dense)

            problems = []
            # [1] and [2]: the reference must agree with the kernel about
            # whether any block was elided. Votes that were not unanimous do
            # not enter this: they are discarded, so they must leave the output
            # bit-identical to dense.
            if not any_skip and differs:
                problems.append(
                    "reference says no block skipped, kernel differs from baseline"
                )
            if any_skip and not differs:
                problems.append("reference says blocks skipped, kernel identical")
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
            print(
                f"{seqlen:>6} {n_q:>3}/{n_kv:<3} {threshold:>10} "
                f"{skipped:>6}/{total:<6} {partial:>8} {str(differs):>8}  "
                f"{verdict}"
            )

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
