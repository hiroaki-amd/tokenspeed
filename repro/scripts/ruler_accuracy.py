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

"""RULER accuracy of skip-softmax against dense, on the same operating point.

Skip-softmax is an approximation, so it costs accuracy, and the honest way to
present it is next to the speedup it buys at the identical threshold and
context length. This script measures both arms in one run: the model is
evaluated once with the stock dense kernel and once with the threshold set, and
the difference is the cost.

Scoring follows RULER's own conventions: recall over all expected strings for
the retrieval and aggregation tasks, partial match for the two QA tasks. The
aggregate is the unweighted mean over tasks, not over samples, so a task with
fewer samples still counts once.

The kernel replaces HF's attention implementation through
``ALL_ATTENTION_FUNCTIONS``. Single-token decode steps fall through to the
original implementation untouched, since this change only affects prefill.

This is the slow half of the bundle. 13 tasks at 50 samples each is 650 prompts
per arm, 1300 in total, and at ctx 32768 that is several hours. Use
``--num-samples`` and ``--tasks`` for a smoke test, and ``--dense-from`` to
reuse a previous run's dense arm rather than recomputing it.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict

import torch

# Inside the container run.sh bind-mounts the RULER tree here.
_DEFAULT_RULER_DATA_DIR = os.environ.get("RULER_DATA") or "/work/ruler"

TASK_TO_CATEGORY = {
    "niah_single_1": "niah",
    "niah_single_2": "niah",
    "niah_single_3": "niah",
    "niah_multikey_1": "niah",
    "niah_multikey_2": "niah",
    "niah_multikey_3": "niah",
    "niah_multivalue": "niah",
    "niah_multiquery": "niah",
    "vt": "variable_tracking",
    "cwe": "common_words_extraction",
    "fwe": "freq_words_extraction",
    "qa_1": "qa",
    "qa_2": "qa",
    # Legacy task names from old generate_ruler_samples.sh
    "niah_single": "niah",
    "niah_multikey": "niah",
}

CATEGORY_MAX_TOKENS = {
    "niah": 128,
    "variable_tracking": 500,
    "common_words_extraction": 500,
    "freq_words_extraction": 500,
    "qa": 128,
}

QA_TASKS = {"qa_1", "qa_2"}


def string_match_all(pred: str, refs: list) -> float:
    """RULER recall scoring: fraction of ALL refs found in prediction."""
    pred_lower = pred.lower()
    return sum(1.0 if str(r).lower() in pred_lower else 0.0 for r in refs) / len(refs)


def string_match_part(pred: str, refs: list) -> float:
    """RULER any-match scoring: 1.0 if ANY ref found in prediction."""
    pred_lower = pred.lower()
    for r in refs:
        if str(r).lower() in pred_lower:
            return 1.0
    return 0.0


def _blasst_attention_fn(threshold: float, original_fn):
    """Build an HF ``ALL_ATTENTION_FUNCTIONS``-compatible callable that routes
    through ``gluon_mha_prefill_gfx950`` instead of a reference Triton kernel.

    Unlike the reference adapter (``attention-evals``' ``blasst_fn``), no
    caller-side ``repeat_kv`` GQA expansion is needed: the kernel derives
    each query head's KV head natively from head-count ratios.

    ``original_fn`` is the attention implementation this replaces (e.g.
    ``sdpa``); single-token decode steps fall through to it unchanged,
    matching ``attention-evals``' ``blasst_fn`` behavior.
    """
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.prefill import (
        gluon_mha_prefill_gfx950,
    )

    def blasst_fn(module, query, key, value, attention_mask, *args, **kwargs):
        seq_len = query.size(-2)
        if seq_len == 1:
            return original_fn(
                module, query, key, value, attention_mask, *args, **kwargs
            )
        if query.size(0) != 1:
            raise RuntimeError(
                "blasst_fn only supports single-sequence (batch=1) prefill calls; "
                f"got query.shape={tuple(query.shape)}"
            )

        q = query[0].transpose(0, 1).contiguous()
        k = key[0].transpose(0, 1).contiguous()
        v = value[0].transpose(0, 1).contiguous()

        cu_seqlens = torch.tensor([0, seq_len], device=q.device, dtype=torch.int32)
        cu_seqlens_cpu = [0, seq_len]

        out = gluon_mha_prefill_gfx950(
            q=q,
            k=k,
            v=v,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            max_seqlen=seq_len,
            skip_softmax_threshold=threshold,
        )

        return out.unsqueeze(0), None

    return blasst_fn


def patch_attention(model, threshold):
    """Replace model's attention with tokenspeed's BLASST kernel."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    attn_impl = model.config._attn_implementation
    original_fn = ALL_ATTENTION_FUNCTIONS[attn_impl]
    ALL_ATTENTION_FUNCTIONS[attn_impl] = _blasst_attention_fn(threshold, original_fn)
    return original_fn, attn_impl


def unpatch_attention(original_fn, attn_impl):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS[attn_impl] = original_fn


def load_ruler_samples(data_dir, ctx_len, task_filter=None, max_samples_per_task=None):
    """Load RULER samples with per-task metadata.

    Args:
        task_filter: If set, only load tasks whose names are in this set.
        max_samples_per_task: If set, cap samples loaded per task.
    """
    ctx_dir = os.path.join(data_dir, f"ctx_{ctx_len}")
    files = sorted(glob.glob(os.path.join(ctx_dir, "*/validation.jsonl")))
    if not files:
        raise FileNotFoundError(f"No validation.jsonl files found in {ctx_dir}/*/")

    samples = []
    task_counts = {}
    for f in files:
        task_name = os.path.basename(os.path.dirname(f))
        if task_filter and task_name not in task_filter:
            continue
        category = TASK_TO_CATEGORY.get(task_name, "niah")
        max_tokens = CATEGORY_MAX_TOKENS.get(category, 128)
        count = 0
        with open(f) as fh:
            for line in fh:
                if max_samples_per_task and count >= max_samples_per_task:
                    break
                entry = json.loads(line)
                samples.append(
                    {
                        "input": entry["input"],
                        "answers": entry["outputs"],
                        "task": task_name,
                        "category": category,
                        "max_new_tokens": max_tokens,
                    }
                )
                count += 1
        task_counts[task_name] = count

    print(f"  Loaded {len(samples)} samples across {len(task_counts)} tasks:")
    for task, count in sorted(task_counts.items()):
        print(f"    {task}: {count} samples")
    return samples


def compute_threshold_from_calibration(
    calibration_json, target_sparsity, context_length
):
    """Compute threshold using paper's formula: lambda = alpha * exp(beta * S) / L."""
    with open(calibration_json) as f:
        data = json.load(f)
    cal = data["calibration"]
    alpha, beta = cal["alpha"], cal["beta"]
    threshold = alpha * math.exp(beta * target_sparsity) / context_length
    print(f"Calibration: alpha={alpha:.4f}, beta={beta:.4f}")
    print(
        f"Formula: lambda = {alpha:.4f} * exp({beta:.4f} * {target_sparsity}) "
        f"/ {context_length}"
    )
    print(f"Computed threshold: lambda = {threshold:.6f}")
    return threshold


def run_evaluation(model, tokenizer, samples, threshold, device):
    """Run model on samples with given threshold, return per-task and aggregate accuracy."""
    if threshold > 0:
        orig_fn, attn_impl = patch_attention(model, threshold)

    task_scores = defaultdict(list)
    total = len(samples)

    for i, sample in enumerate(samples):
        messages = [{"role": "user", "content": sample["input"]}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer.encode(text, return_tensors="pt").to(device)
        max_new_tokens = sample.get("max_new_tokens", 128)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True)

        task_name = sample["task"]
        if task_name in QA_TASKS:
            score = string_match_part(generated, sample["answers"])
        else:
            score = string_match_all(generated, sample["answers"])

        task_scores[task_name].append(score)

        if (i + 1) % 50 == 0:
            running_acc = sum(s for scores in task_scores.values() for s in scores) / (
                i + 1
            )
            print(f"  [{i+1}/{total}] running avg: {running_acc*100:.1f}%", flush=True)

    if threshold > 0:
        unpatch_attention(orig_fn, attn_impl)

    per_task = {}
    for task_name, scores in sorted(task_scores.items()):
        per_task[task_name] = sum(scores) / len(scores) * 100

    aggregate = sum(per_task.values()) / len(per_task) if per_task else 0.0
    return aggregate, per_task


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--data-dir", default=_DEFAULT_RULER_DATA_DIR)
    p.add_argument("--context-length", type=int, default=32768)
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed BLASST threshold (legacy mode)",
    )
    p.add_argument(
        "--target-sparsity",
        type=float,
        default=None,
        help="Target sparsity (0-1). Computes threshold via calibration formula.",
    )
    p.add_argument(
        "--calibration-json",
        default=None,
        help="Path to calibration JSON (required with --target-sparsity)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-file", default=None, help="Write results to this file")
    p.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task names to evaluate (default: all)",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Max samples per task (default: all)",
    )
    p.add_argument(
        "--dense-from",
        default=None,
        help=(
            "Reuse the dense baseline from a previous run's output JSON instead of "
            "recomputing it. Only valid when that run used the same model, context "
            "length, task set and sample count (checked below)."
        ),
    )
    args = p.parse_args()

    if args.target_sparsity is not None:
        if args.calibration_json is None:
            p.error("--calibration-json is required when using --target-sparsity")
        threshold = compute_threshold_from_calibration(
            args.calibration_json, args.target_sparsity, args.context_length
        )
    elif args.threshold is not None:
        threshold = args.threshold
    else:
        p.error("Either --threshold or --target-sparsity must be specified")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()

    task_filter = set(args.tasks.split(",")) if args.tasks else None
    samples = load_ruler_samples(
        args.data_dir,
        args.context_length,
        task_filter=task_filter,
        max_samples_per_task=args.num_samples,
    )
    print(f"Loaded {len(samples)} total RULER samples at ctx={args.context_length}")

    if args.dense_from:
        # The dense baseline is threshold-independent, so a second threshold on the
        # same (model, ctx, tasks, sample count) can reuse it rather than spending
        # another full pass on it. Refuse to reuse across a different sample set,
        # since the per-task numbers would not be comparable.
        with open(args.dense_from) as f:
            prev = json.load(f)
        if prev["context_length"] != args.context_length or prev["num_samples"] != len(
            samples
        ):
            raise SystemExit(
                f"--dense-from {args.dense_from} was run on ctx="
                f"{prev['context_length']}/{prev['num_samples']} samples, but this run "
                f"is ctx={args.context_length}/{len(samples)} samples; refusing to reuse."
            )
        dense_agg = prev["dense"]["aggregate"]
        dense_per_task = prev["dense"]["per_task"]
        dense_time = prev["dense"]["time_s"]
        print(f"\n--- Dense (threshold=0) [reused from {args.dense_from}] ---")
        print(f"Dense aggregate accuracy: {dense_agg:.2f}%")
    else:
        print("\n--- Dense (threshold=0) ---")
        t0 = time.time()
        dense_agg, dense_per_task = run_evaluation(
            model, tokenizer, samples, 0.0, args.device
        )
        dense_time = time.time() - t0
        print(f"Dense aggregate accuracy: {dense_agg:.2f}% ({dense_time:.0f}s)")
    print("Per-task:")
    for task, acc in sorted(dense_per_task.items()):
        print(f"  {task}: {acc:.1f}%")

    print(f"\n--- BLASST (threshold={threshold:.6f}) ---")
    if args.target_sparsity is not None:
        print(f"  Target sparsity: {args.target_sparsity*100:.0f}%")
    t0 = time.time()
    blasst_agg, blasst_per_task = run_evaluation(
        model, tokenizer, samples, threshold, args.device
    )
    blasst_time = time.time() - t0
    print(f"BLASST aggregate accuracy: {blasst_agg:.2f}% ({blasst_time:.0f}s)")
    print("Per-task:")
    for task, acc in sorted(blasst_per_task.items()):
        print(f"  {task}: {acc:.1f}%")

    print(f"\n{'='*60}")
    print(
        f"RESULTS SUMMARY: ctx={args.context_length}, {len(samples)} samples, "
        f"{len(dense_per_task)} tasks"
    )
    print(f"{'='*60}")
    print(f"{'Task':<25} {'Dense':>8} {'BLASST':>8} {'Drop':>8}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for task in sorted(dense_per_task.keys()):
        d = dense_per_task[task]
        b = blasst_per_task.get(task, 0)
        print(f"{task:<25} {d:>7.1f}% {b:>7.1f}% {b-d:>+7.1f}%")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    print(
        f"{'AGGREGATE':<25} {dense_agg:>7.2f}% {blasst_agg:>7.2f}% {blasst_agg-dense_agg:>+7.2f}%"
    )

    if args.output_file:
        results = {
            "context_length": args.context_length,
            "num_samples": len(samples),
            "num_tasks": len(dense_per_task),
            "threshold": threshold,
            "target_sparsity": args.target_sparsity,
            "dense": {
                "aggregate": dense_agg,
                "per_task": dense_per_task,
                "time_s": dense_time,
            },
            "blasst": {
                "aggregate": blasst_agg,
                "per_task": blasst_per_task,
                "time_s": blasst_time,
            },
        }
        out_dir = os.path.dirname(args.output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: GPU required.")
        sys.exit(1)
    main()
