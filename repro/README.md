# Skip-softmax MHA prefill: reproduction bundle

Everything needed to re-measure the numbers in the skip-softmax PR, in a
container so the toolchain is not a variable.

The change adds BLASST-style block sparsity (arXiv:2512.12087) to tokenspeed's
Gluon MHA prefill kernel on gfx950 (MI350X): a query row votes to skip a K/V
block when its contribution to the softmax is negligible against the running
max, and the block is dropped when every row of the query tile votes for it.
Three things ship together, and the bundle measures all three,
because the first two on their own make real workloads *slower*:

1. `skip_softmax_threshold`, the skip itself.
2. `defer_v_load`, so a fully-skipped block costs no V traffic either.
3. A global atomic work counter replacing the static workgroup-to-head
   assignment. Block sparsity varies sharply between heads of the same layer,
   and with one head pinned per workgroup a layer runs at close to worst-head
   speed.

This directory lives on the branch that carries the change itself
(`feat/mha-skip-softmax-dynamic-sched`), rather than as a separate orphan
branch, so the kernel the image builds is always the one checked out here:
there is no separate checkout to point at or fall out of sync with.

## What you need

- An MI350X (gfx950). The kernel dispatches on that target and will not run
  elsewhere.
- Docker, with access to `/dev/kfd` and `/dev/dri`. Your user must be in the
  `video` and `render` groups.
- This repository checked out at the commit you want to measure. The image
  bakes in `tokenspeed-kernel-amd` from the build context, so a kernel change
  needs a rebuild. `docker/run.sh` handles that on its own: the default image
  tag ends in a hash of the kernel tree, so a changed kernel is a different tag
  and rebuilds, and an older kernel you go back to still has its image.
- About 20 GB of disk for the image, plus 16 GB for the Qwen3-8B weights.
- RULER task data, for the real-activation and accuracy runs, laid out as
  `<RULER_DATA>/ctx_<length>/<task>/validation.jsonl`. The synthetic sweep needs
  neither RULER nor a model and is the fastest way to see something work.

## Running it

From the repository root:

```
export RULER_DATA=/path/to/ruler        # only for stages 2 and 3

./repro/docker/run.sh bash scripts/all.sh                 # everything
QUICK=1 ./repro/docker/run.sh bash scripts/all.sh         # smoke test, ~20 min
SKIP_ACCURACY=1 ./repro/docker/run.sh bash scripts/all.sh # skip the slow half
./repro/docker/run.sh                                     # interactive shell
```

The first invocation builds the image, which takes several minutes. Results are
written to `results/` as a log and a JSON file per stage. `RESULTS.md` has the
numbers this produced on our MI350X, to compare against.

Budget roughly: synthetic sweep 10 minutes, RULER speed 40 minutes, RULER
accuracy several hours (650 prompts per arm, two arms).

## The stages

**`scripts/sparsity_sweep.py`** sweeps threshold against sparsity and speedup on
random Q/K/V, at two GQA shapes. Random data makes every head equally sparse,
which removes the imbalance the work counter exists to fix, so these numbers are
a floor for the scheduling change rather than a showcase. The two shapes are
64/4 (GQA ratio 16, the shape in the BLASST paper's Table 5) and 32/8 (ratio 4,
Qwen3-8B's shape). The head-interleave group size is derived from the GQA ratio,
so the two shapes exercise interleave groups of 16 and 4 respectively.

**`scripts/ruler_speed.py`** captures per-layer Q/K/V from a real Qwen3-8B
forward pass over each RULER task and times the kernel on those tensors. This is
the measurement that matters, and the one the third component was written for.

**`scripts/ruler_accuracy.py`** runs the full RULER evaluation twice, once dense
and once with the threshold set, and reports the difference. Skip-softmax is an
approximation and this is what it costs.

**`scripts/verify_reference.py`** checks the sparsity counter (below) against
the kernel's observable behaviour. `all.sh` runs it before the three stages, as
stage 0, and it takes well under a minute. A failure does not stop the run,
since speed and accuracy never go through the reference, but it exits nonzero
and says so again at the end: the sparsity columns are then not to be quoted.

## About the sparsity numbers

The kernel does not report sparsity. Counters would need atomics on the hot
path, so they were left out of the change deliberately. Sparsity here is
recomputed independently in PyTorch by `scripts/sparsity_reference.py`, which
is the better arrangement for a reproduction bundle anyway: a number that can
only be produced by the thing under test cannot corroborate it.

That reference was checked against an instrumented build of the kernel that did
carry counters, and matched exactly (not approximately) on five shapes.
`verify_reference.py` re-runs a weaker version of that check that needs no
instrumented build, testing the reference's predictions against the kernel's
output instead: no rows skipped implies bit-identical output, any rows skipped
implies changed output, sparsity monotonic in the threshold, and `defer_v_load`
never changing the result. 30 checks across 5 shapes and 6 thresholds.

The counter is still `O(seqlen^2)`, but it batches the head and query-tile axes
into GPU ops rather than looping over them in Python (only the KV-block axis is
a Python loop, since that is where the running-max recurrence lives). That
brings a 16k count down to well under a second and a 64k count to a few
seconds, so `scripts/sparsity_sweep.py` computes it unconditionally at every
length it sweeps, including 64k, and `scripts/ruler_speed.py` no longer needs
`--no-sparsity` at ctx 32768 either.

Two caveats worth reading before quoting a sparsity figure.

**The row-level rate runs far ahead of the block-level one.** A block counts as
sparsity only when *all* of its query rows vote to skip it, because that is the
only case where anything is elided. Where some rows vote and others do not, the
votes are discarded and every row takes the ordinary update, so the block costs
full time and leaves the result unchanged. There is a wide band of thresholds
where rows vote in quantity and sparsity is still zero, and across that band the
output is bit-identical to dense.

**Replay sparsity reads higher than live sparsity.** `ruler_speed.py` captures
activations under dense attention and replays them. In a real run the
approximation perturbs each layer's input, and sparsity comes out lower. Timings
are unaffected, since they depend only on the tensors passed in, but the
sparsity column in the speed run is replay sparsity and should be labelled as
such.

## Calibration

The threshold is a probability ratio, not a score, and the sparsity it produces
depends on the score distribution. That shifts with sequence length and with the
model, so it has to be calibrated per workload. `0.03` at ctx 32768 on Qwen3-8B
drops roughly 60% of blocks live; the same threshold means something different
elsewhere. `skip_softmax_threshold=0.0` is the default and is exact dense
attention, bit-identical to the kernel without the feature.

## Toolchain

The image pins ROCm 7.2.4, torch 2.11.0+rocm7.2 and tokenspeed-triton
3.8.10.post20260721. The Gluon dialect the kernel is written in lives in
tokenspeed's Triton fork, which installs as its own `tokenspeed_triton` package,
so whichever stock `triton` the base image happens to carry does not matter.

The same synthetic sweep was also run on ROCm 7.0 with torch 2.10, and agreed to
within 0.006x, so the results are not knife-edge dependent on those pins.
