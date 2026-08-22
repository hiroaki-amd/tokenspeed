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

"""Per-layer skip-softmax prefill speedup on real Qwen3-8B activations.

Real text is the measurement that matters for this change, and it is the one
that motivated the scheduling work. Capture per-layer Q/K/V from a genuine
Qwen3-8B forward pass over a RULER prompt (via HF's ``ALL_ATTENTION_FUNCTIONS``
hook), then time ``gluon_mha_prefill_gfx950`` directly on each layer's captured
tensors, and sum over the model's layers.

Why per-layer rather than one wall-clock number: a full ``model.generate()``
mixes every layer's prefill attention together with decode steps this change
does not touch, and per-layer sparsity at a fixed threshold varies enormously,
so an aggregate hides exactly the structure that determines whether the kernel
wins.

Three configurations are timed per layer:

  dense       ``skip_softmax_threshold=0.0``, the stock kernel, which is also
              the static schedule. This is the baseline a user has today.
  skip only   threshold set, ``defer_v_load=False``.
  V-deferred  threshold set, ``defer_v_load=True``.

In the merged API a nonzero threshold also switches on the dynamic work
counter, so neither of the latter two isolates the scheduler. That is by
design: the counter costs about 2% when there is no sparsity to earn it back,
so it is not something a caller should be able to enable on its own.

A task's speedup is its layer times summed and then divided, so each layer is
weighted by how long it takes. The reported mean is the mean over tasks of
that. Averaging the per-layer ratios instead gives the cheap early layers, which
have almost no sparsity to exploit, the same weight as the expensive ones, and
reads about 0.01x higher. Both are in the JSON, as ``v_deferred_speedup`` and
``mean_v_deferred_speedup``.

One caveat that is easy to get wrong when reading these numbers. Activations
are captured under *dense* attention and then replayed, so the sparsity
reported here is what dense-derived scores imply. In a real run the
approximation perturbs each layer's input, and measured sparsity comes out
lower (43.9% live versus 54.9% replayed, at threshold 0.03 and ctx 32768). The
timings are unaffected, since they only depend on the tensors actually passed
in, but the sparsity column should be read as replay sparsity.

Sparsity is recomputed independently in PyTorch by ``sparsity_reference``
rather than read out of the kernel, which has no counters. It is O(seqlen^2)
and slow; pass ``--no-sparsity`` for timings alone.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch
from sparsity_reference import count_block_sparsity

# Inside the container run.sh bind-mounts the RULER tree here.
_DEFAULT_RULER_DATA_DIR = os.environ.get("RULER_DATA") or "/work/ruler"

ALL_TASKS = [
    "cwe",
    "fwe",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "qa_1",
    "qa_2",
    "vt",
]


def load_one_ruler_sample(data_dir, task, ctx_len):
    """Return the ``input`` text of the first RULER sample for task/ctx_len."""
    pattern = os.path.join(data_dir, f"ctx_{ctx_len}", task, "validation.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No validation.jsonl found at {pattern}")
    with open(files[0]) as fh:
        return json.loads(fh.readline())["input"]


def build_model(model_name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    return tokenizer, model


def capture_model_qkv(tokenizer, model, text, device):
    """Run one forward pass and capture packed Q/K/V per layer.

    Returns ``{layer_idx: {"Q": (S,Hq,D), "K": (S,Hkv,D), "V": (S,Hkv,D)}}``
    already squeezed out of the batch=1 dim, matching the
    ``[total_tokens, n_heads, head_dim]`` layout ``gluon_mha_prefill_gfx950``
    expects directly (no ``repeat_kv`` needed -- the kernel derives GQA
    natively from head-count ratios).
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    impl = model.config._attn_implementation
    original_fn = ALL_ATTENTION_FUNCTIONS[impl]
    captured = {}

    def hook(module, query, key, value, attention_mask, *args, **kwargs):
        captured[module.layer_idx] = {
            "Q": query[0].transpose(0, 1).contiguous().detach(),
            "K": key[0].transpose(0, 1).contiguous().detach(),
            "V": value[0].transpose(0, 1).contiguous().detach(),
        }
        return original_fn(module, query, key, value, attention_mask, *args, **kwargs)

    ALL_ATTENTION_FUNCTIONS[impl] = hook
    try:
        # Match the accuracy eval's chat-template wrapping so the captured
        # activations correspond to the prompts that produced its numbers.
        messages = [{"role": "user", "content": text}]
        chat = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tokenizer.encode(chat, return_tensors="pt").to(device)
        with torch.no_grad():
            model(ids)
    finally:
        ALL_ATTENTION_FUNCTIONS[impl] = original_fn
    torch.cuda.empty_cache()
    return captured


def benchmark_fn(fn, warmup, repeat):
    """Mean wall-clock ms per call, after warmup, bracketed by device syncs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / repeat * 1000.0


def bench_one_layer(qkv, threshold, warmup, repeat, measure_sparsity=True):
    """Time dense / skip-only / V-deferred for one layer, and measure sparsity.

    Args:
        qkv: dict with captured ``Q``/``K``/``V`` for one layer, packed
            ``[seqlen, n_heads, head_dim]``.
        threshold: ``skip_softmax_threshold`` to evaluate.
        warmup: untimed calls before timing.
        repeat: timed calls to average over.
        measure_sparsity: recompute block sparsity in PyTorch. This is
            O(seqlen^2) and dominates the runtime at long contexts, so it can
            be turned off when only the timings are wanted.

    Returns:
        A dict with ``sparsity_pct`` (None when not measured), and the dense,
        skip-only and V-deferred milliseconds with their speedups over dense.
    """
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.prefill import (
        gluon_mha_prefill_gfx950,
    )

    q, k, v = qkv["Q"], qkv["K"], qkv["V"]
    seq_len = q.shape[0]
    cu_seqlens = torch.tensor([0, seq_len], device=q.device, dtype=torch.int32)
    cu_seqlens_cpu = [0, seq_len]

    def call(th, defer_v_load):
        return gluon_mha_prefill_gfx950(
            q=q,
            k=k,
            v=v,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            max_seqlen=seq_len,
            skip_softmax_threshold=th,
            defer_v_load=defer_v_load,
        )

    if measure_sparsity:
        total_n, skipped_n = count_block_sparsity(q, k, threshold)
        sparsity = 100.0 * skipped_n / total_n if total_n else 0.0
    else:
        sparsity = None

    dense_ms = benchmark_fn(lambda: call(0.0, False), warmup, repeat)
    skip_ms = benchmark_fn(lambda: call(threshold, False), warmup, repeat)
    defer_ms = benchmark_fn(lambda: call(threshold, True), warmup, repeat)

    return {
        "sparsity_pct": sparsity,
        "dense_ms": dense_ms,
        "skip_only_ms": skip_ms,
        "skip_only_speedup": dense_ms / skip_ms if skip_ms else 0.0,
        "v_deferred_ms": defer_ms,
        "v_deferred_speedup": dense_ms / defer_ms if defer_ms else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--data-dir", default=_DEFAULT_RULER_DATA_DIR)
    p.add_argument("--context-length", type=int, default=32768)
    p.add_argument("--threshold", type=float, default=0.03)
    p.add_argument("--tasks", default=None, help="Comma-separated (default: all 13)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-file", default=None)
    p.add_argument(
        "--no-sparsity",
        action="store_true",
        help="Skip the O(seqlen^2) PyTorch sparsity recomputation.",
    )
    args = p.parse_args()

    tasks = args.tasks.split(",") if args.tasks else ALL_TASKS

    print(f"Loading {args.model}...")
    tokenizer, model = build_model(args.model, args.device)

    results = {}
    for task in tasks:
        text = load_one_ruler_sample(args.data_dir, task, args.context_length)
        captured = capture_model_qkv(tokenizer, model, text, args.device)
        layers = sorted(captured)

        per_layer = {}
        for li in layers:
            per_layer[li] = bench_one_layer(
                captured[li],
                args.threshold,
                args.warmup,
                args.repeat,
                measure_sparsity=not args.no_sparsity,
            )

        del captured
        torch.cuda.empty_cache()

        n = len(per_layer)
        # Two aggregations, and they are not the same number. Summing the
        # per-layer times and dividing once weights each layer by how long it
        # actually takes, which is what a user experiences and what the PR
        # quotes. Averaging the per-layer ratios instead gives every layer
        # equal say, including the cheap early ones that have almost no
        # sparsity to exploit, and reads about 0.01x high as a result. The
        # summed figure is the headline; the ratio mean is kept because the
        # per-layer distribution is the thing the scheduling change is about.
        tot_dense = sum(r["dense_ms"] for r in per_layer.values())
        tot_skip = sum(r["skip_only_ms"] for r in per_layer.values())
        tot_defer = sum(r["v_deferred_ms"] for r in per_layer.values())
        skip_speedup = tot_dense / tot_skip if tot_skip else 0.0
        defer_speedup = tot_dense / tot_defer if tot_defer else 0.0
        mean_skip = sum(r["skip_only_speedup"] for r in per_layer.values()) / n
        mean_defer = sum(r["v_deferred_speedup"] for r in per_layer.values()) / n
        sps = [
            r["sparsity_pct"]
            for r in per_layer.values()
            if r["sparsity_pct"] is not None
        ]
        mean_sp = sum(sps) / len(sps) if sps else None
        results[task] = {
            "mean_sparsity_pct": mean_sp,
            "total_dense_ms": tot_dense,
            "total_skip_only_ms": tot_skip,
            "total_v_deferred_ms": tot_defer,
            "skip_only_speedup": skip_speedup,
            "v_deferred_speedup": defer_speedup,
            "mean_skip_only_speedup": mean_skip,
            "mean_v_deferred_speedup": mean_defer,
            "layers_above_1x_v_deferred": sum(
                1 for r in per_layer.values() if r["v_deferred_speedup"] >= 1.0
            ),
            "num_layers": n,
            "per_layer": {str(li): r for li, r in per_layer.items()},
        }
        sp_txt = f"{mean_sp:.1f}%" if mean_sp is not None else "n/a"
        print(
            f"  {task}: sparsity {sp_txt}, skip-only {skip_speedup:.3f}x, "
            f"V-deferred {defer_speedup:.3f}x "
            f"({results[task]['layers_above_1x_v_deferred']}/{n} layers >= 1.0x)",
            flush=True,
        )

    print(f"\n{'='*78}")
    print(
        f"PER-LAYER SPEEDUP on real activations: ctx={args.context_length}, "
        f"threshold={args.threshold}"
    )
    print(f"{'='*78}")
    hdr = (
        f"{'Task':<20}{'sparsity':>10}{'skip-only':>12}{'V-deferred':>13}{'>=1.0x':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for task in tasks:
        r = results[task]
        sp = r["mean_sparsity_pct"]
        sp_txt = f"{sp:.1f}%" if sp is not None else "n/a"
        print(
            f"{task:<20}{sp_txt:>10}{r['skip_only_speedup']:>11.3f}x"
            f"{r['v_deferred_speedup']:>12.3f}x"
            f"{r['layers_above_1x_v_deferred']:>5}/{r['num_layers']:<3}"
        )
    print("-" * len(hdr))
    sps = [
        results[t]["mean_sparsity_pct"]
        for t in tasks
        if results[t]["mean_sparsity_pct"] is not None
    ]
    m_sp = f"{sum(sps) / len(sps):.1f}%" if sps else "n/a"
    m_sk = sum(results[t]["skip_only_speedup"] for t in tasks) / len(tasks)
    m_df = sum(results[t]["v_deferred_speedup"] for t in tasks) / len(tasks)
    print(f"{'MEAN':<20}{m_sp:>10}{m_sk:>11.3f}x{m_df:>12.3f}x")
    print(
        "\nSpeedups are per task: layer times summed, then divided. MEAN is the "
        "mean over tasks of that.\nAveraging the per-layer ratios instead reads "
        "about 0.01x higher; both are in the JSON."
    )

    if args.output_file:
        out_dir = os.path.dirname(args.output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "context_length": args.context_length,
                    "threshold": args.threshold,
                    "warmup": args.warmup,
                    "repeat": args.repeat,
                    "per_task": results,
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: GPU required.")
        sys.exit(1)
    main()
