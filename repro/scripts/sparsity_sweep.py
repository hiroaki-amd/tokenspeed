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

"""Threshold-to-sparsity-to-speedup sweep on synthetic Q/K/V.

This is the measurement behind the synthetic table in the PR. Random Q/K/V is
not a realistic workload, and the point of including it is not to claim a
speedup on real text; it is that random data makes every attention head equally
sparse. That removes the head imbalance the dynamic work counter exists to fix,
so whatever the counter costs here, it cannot earn back. These numbers are a
floor for the scheduling change rather than a showcase for it.

Two GQA shapes are swept because the head-interleave group size is derived from
the GQA ratio rather than tuned, and the two ratios are far apart:

    64 Q-heads / 4 KV-heads   ratio 16, the shape used in the BLASST paper's
                              Table 5, and the widest interleave group the
                              derivation produces
    32 Q-heads / 8 KV-heads   ratio 4, the shape of Qwen3-8B, so it matches the
                              real-activation measurement reported alongside

Sparsity is not read out of the kernel. The kernel has no counters (they would
need atomics on the hot path), so it is recomputed independently in PyTorch by
``sparsity_reference``, which was verified to agree exactly with an
instrumented build of the kernel. A number that can only be produced by the
thing under test cannot corroborate it.

The reference counter is still O(seqlen^2), but ``sparsity_reference`` batches
the head and query-tile axes into GPU ops instead of looping over them in
Python, so it is cheap enough (a couple of seconds per threshold at 64k) to
compute unconditionally at every sequence length swept here.

Thresholds are chosen to span roughly 0% to 95% sparsity at each length, the
same range as Table 5 of the BLASST paper, rather than to cluster around 50%:
comparing against that table needs the same coverage of the curve, not just
its midpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from sparsity_reference import count_block_sparsity
from tokenspeed_kernel_amd.ops.gfx950.attention.mha.prefill import (
    gluon_mha_prefill_gfx950,
)

HEAD_DIM = 128
DTYPE = torch.bfloat16

SHAPES = [
    (64, 4, "paper Table 5 shape"),
    (32, 8, "Qwen3-8B shape"),
]

# Thresholds chosen to span ~0% to ~95% sparsity at each length, matching the
# range of BLASST paper Table 5. Sparsity at a fixed threshold shifts with
# sequence length, which is exactly why the values differ between the two
# rows below.
SWEEP = {
    16384: [1e-9, 1.0, 1.3, 1.7, 4.0, 6.0, 8.0, 10.0],
    65536: [1e-9, 0.7, 0.8, 0.9, 1.1, 2.0, 6.0, 10.0],
}


def make_qkv(seqlen: int, n_q: int, n_kv: int, device: str):
    """Build one causal batch of random Q/K/V, seeded by sequence length."""
    torch.manual_seed(seqlen)
    q = torch.randn(seqlen, n_q, HEAD_DIM, device=device, dtype=DTYPE)
    k = torch.randn(seqlen, n_kv, HEAD_DIM, device=device, dtype=DTYPE)
    v = torch.randn(seqlen, n_kv, HEAD_DIM, device=device, dtype=DTYPE)
    return q, k, v


def benchmark_fn(fn, warmup: int, repeat: int) -> float:
    """Return the mean wall-clock milliseconds of ``fn`` after warmup."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / repeat * 1000.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-file", default=None)
    args = p.parse_args()

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch {torch.__version__}")
    print(f"head_dim={HEAD_DIM} dtype={DTYPE} causal=True batch=1\n")

    results = []
    for n_q, n_kv, note in SHAPES:
        print(
            f"=== {n_q} Q-heads / {n_kv} KV-heads, GQA ratio {n_q // n_kv} "
            f"({note}) ==="
        )
        for seqlen, thresholds in SWEEP.items():
            q, k, v = make_qkv(seqlen, n_q, n_kv, args.device)
            cu = torch.tensor([0, seqlen], device=args.device, dtype=torch.int32)

            def call(threshold, defer):
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

            # skip_softmax_threshold=0.0 is the stock dense kernel: no skip
            # check, no deferred V load, and the static scheduler. That is the
            # right baseline, because it is what a user gets today.
            dense_ms = benchmark_fn(lambda: call(0.0, False), args.warmup, args.repeat)

            print(
                f"\n  seqlen={seqlen} ({seqlen // 1024}k)   " f"dense {dense_ms:.3f} ms"
            )
            header = (
                f"  {'threshold':>10} {'sparsity':>10} {'blasst ms':>11} {'speedup':>9}"
            )
            print(header)

            for threshold in thresholds:
                ms = benchmark_fn(
                    lambda: call(threshold, True), args.warmup, args.repeat
                )
                total, skipped = count_block_sparsity(q, k, threshold)
                pct = 100.0 * skipped / max(total, 1)
                sp = f"{pct:.2f}%"
                print(
                    f"  {threshold:>10} {sp:>10} {ms:>11.3f} " f"{dense_ms / ms:>8.3f}x"
                )
                results.append(
                    {
                        "n_q_heads": n_q,
                        "n_kv_heads": n_kv,
                        "gqa_ratio": n_q // n_kv,
                        "seqlen": seqlen,
                        "threshold": threshold,
                        "sparsity_pct": pct,
                        "dense_ms": dense_ms,
                        "blasst_ms": ms,
                        "speedup": dense_ms / ms,
                    }
                )

            del q, k, v
            torch.cuda.empty_cache()
        print()

    if args.output_file:
        d = os.path.dirname(args.output_file)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.output_file, "w") as fh:
            json.dump(
                {
                    "device": torch.cuda.get_device_name(0),
                    "torch": torch.__version__,
                    "rows": results,
                },
                fh,
                indent=2,
            )
        print(f"wrote {args.output_file}")


if __name__ == "__main__":
    main()
