# Reference numbers

What this bundle produced on MI350X, for comparison against your own run. All
of it was measured through `docker/run.sh`, against kernel commit `073f4353`
("perf(amd): schedule skip-softmax MHA prefill from a dynamic work counter").

That commit was later amended to `7daa6f67` to pick up a black reformat of two
call sites in a test file, so `073f4353` is no longer on the branch. The kernel
is byte for byte the same, and these numbers apply unchanged to `7daa6f67` and
to this directory, which lives on that same branch.

The raw logs and the per-layer JSON behind every table below are committed in
`results/`; see `results/README.md` for what each file is and which ones not to
quote.

## Stage 1: synthetic sparsity sweep

`scripts/sparsity_sweep.py`, random Q/K/V, causal, head_dim=128, bf16, batch 1,
mean of 20 timed repeats after 5 warmups, dense measured in the same run.

```
=== 64 Q-heads / 4 KV-heads, GQA ratio 16 (paper Table 5 shape) ===

  seqlen=16384 (16k)   dense 8.438 ms
   threshold   sparsity   blasst ms   speedup
         0.9     13.25%       8.637    0.977x
         1.1     29.10%       8.063    1.047x
         1.3     39.13%       7.779    1.085x
         1.7     49.98%       7.474    1.129x
         2.0     54.70%       7.355    1.147x

  seqlen=65536 (64k)   dense 136.791 ms
   threshold   sparsity   blasst ms   speedup
         0.7          -     130.018    1.052x
         0.8          -     122.968    1.112x
         0.9          -     117.117    1.168x
         1.0          -     112.739    1.213x
         1.1          -     110.031    1.243x

=== 32 Q-heads / 8 KV-heads, GQA ratio 4 (Qwen3-8B shape) ===

  seqlen=16384 (16k)   dense 4.228 ms
   threshold   sparsity   blasst ms   speedup
         0.9     13.21%       4.417    0.957x
         1.1     29.09%       4.122    1.026x
         1.3     39.14%       3.968    1.066x
         1.7     50.17%       3.816    1.108x
         2.0     54.88%       3.758    1.125x

  seqlen=65536 (64k)   dense 69.539 ms
   threshold   sparsity   blasst ms   speedup
         0.7          -      65.250    1.066x
         0.8          -      61.967    1.122x
         0.9          -      59.316    1.172x
         1.0          -      57.264    1.214x
         1.1          -      55.907    1.244x
```

The sparsity column is blank at 64k because the reference counter is a Python
loop over blocks and is `O(n^2)`; `--sparsity-all` computes it anyway. Sparsity
does not depend on the toolchain, so the 16k column carries the same
information.

Against the same sweep run outside the container, 18 of 20 rows agree to within
0.002x. The two exceptions are 32/8 at 16k with thresholds 0.9 and 1.1
(+0.007x and +0.003x), which are the two lowest-sparsity points of the smallest
shape, where the measurement is a small difference between two nearly equal
times. The same
sweep on a ROCm 7.0 / torch 2.10 toolchain agreed to within 0.006x, so the
numbers are not sensitive to the exact pins.

## Stage 2: RULER per-layer speed

`scripts/ruler_speed.py`, Qwen3-8B, per-layer attention time summed over 36
layers, dense measured in the same run.

Full run, all 13 tasks, ctx=32768, threshold 0.03, mean over tasks:

```
  configuration                                speedup
  dense                                         1.000x
  skip-softmax only                             0.980x
  + defer_v_load                                0.942x
  + dynamic work counter                        1.135x
```

The middle two rows are the reason the work counter ships with the feature and
not after it. Skipping the matmul does not pay for the check on its own, and
deferring V makes it worse, because both are averaged away by whichever head in
the layer is least sparse. Those two rows predate the counter and cannot be
reproduced from this bundle, because a nonzero threshold now switches the
counter on unconditionally. They are quoted from the earlier measurement.

The last row is what this bundle produces. Per task, against the same
measurement taken outside the container:

```
Task                  bundle   earlier     diff
cwe                   1.092x    1.090x   +0.002
fwe                   1.207x    1.202x   +0.005
niah_multikey_1       1.153x    1.151x   +0.002
niah_multikey_2       1.062x    1.060x   +0.003
niah_multikey_3       1.096x    1.094x   +0.002
niah_multiquery       1.156x    1.153x   +0.003
niah_multivalue       1.117x    1.114x   +0.002
niah_single_1         1.132x    1.131x   +0.001
niah_single_2         1.117x    1.116x   +0.001
niah_single_3         1.116x    1.115x   +0.002
qa_1                  1.145x    1.143x   +0.001
qa_2                  1.250x    1.246x   +0.003
vt                    1.113x    1.113x   +0.000
MEAN                  1.135x    1.133x   +0.002
```

Every task agrees to within 0.005x and the container reads uniformly very
slightly faster, which is the shape of a small systematic offset rather than
noise.

Note the aggregation. A task's speedup is its layer times summed and then
divided, so each layer counts for as much time as it takes. Averaging the
per-layer ratios instead gives the cheap early layers, which have little
sparsity to exploit, equal weight with the expensive ones, and reads about
0.01x higher (1.143x here). Both are in the JSON. Quoting the two
interchangeably is an easy way to produce a spurious disagreement.

`QUICK=1`, 2 tasks, ctx=8192, threshold 0.03, as a smoke test. Speedups are
lower than at 32768 because attention is a smaller share of the work at 8k, so
do not compare these against the table above:

```
Task                  sparsity   skip-only   V-deferred   >=1.0x
niah_single_1            44.8%      1.064x       1.084x   33/36
qa_1                     42.6%      1.063x       1.092x   29/36
MEAN                     43.7%      1.063x       1.088x
```

These were produced before the aggregation fix above and are per-layer ratio
means, so they read about 0.01x high.

Note the sparsity column here is replay sparsity and reads high. See the
caveats in README.md.

## Stage 3: RULER accuracy

`scripts/ruler_accuracy.py`, Qwen3-8B, ctx=32768, threshold 0.03, 13 tasks x 50
samples (650 prompts per arm):

```
  13-task aggregate    91.58%  ->  90.30%    (-1.28 pts)
```

7 of 13 tasks degrade. The failure mode is aggregation over many positions
(fwe -5.3, cwe -4.4), not pinpoint retrieval: all three `niah_single` tasks
hold at 100.0%.

`QUICK=1` is 3 samples per task and is far too small to say anything about
accuracy. It exists to prove the pipeline runs. For the record it came out at
100.0% / 33.3% on both arms, dense and BLASST alike.

## Sparsity

Live block sparsity at threshold 0.03 and ctx 32768 is 43.9%. The replayed
figure on the same prompts is 54.9%. The live figure is the one that describes
a real run; see README.md for why they differ.
