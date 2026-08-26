# Reference numbers

> **Stale: these predate `7d742874` and do not describe the kernel this bundle
> now builds.** That commit changed the skip rule from per row to per block. The
> per-row test now only casts a vote, and a block is dropped only when every row
> of the query tile agrees; previously a dissenting block still had the voting
> rows' softmax numerator zeroed. Holding a skipped row's running max back made
> that row less likely to clear the threshold on later blocks, so the old rule
> cost accuracy and sparsity both. At a fixed threshold the numbers below
> therefore understate sparsity and speed while overstating the accuracy loss:
> RULER moves from 91.58% -> 90.30% (-1.28 pts) to 91.24% -> 91.52% (+0.28 pts),
> and block sparsity at threshold 0.03 from 43.9% to 59.6%. Re-run the bundle
> for current figures. This file is kept only so that a run against the old
> kernel stays interpretable.

What this bundle produced on MI350X, for comparison against your own run. All
of it was measured through `docker/run.sh`, against kernel commit `073f4353`
("perf(amd): schedule skip-softmax MHA prefill from a dynamic work counter").

That commit was later amended to `7daa6f67` to pick up a black reformat of two
call sites in a test file, so `073f4353` is no longer on the branch. The kernel
is byte for byte the same, and these numbers apply unchanged to `7daa6f67`.

The raw logs and the per-layer JSON behind every table below are committed in
`results/`; see `results/README.md` for what each file is and which ones not to
quote.

## Stage 1: synthetic sparsity sweep

`scripts/sparsity_sweep.py`, random Q/K/V, causal, head_dim=128, bf16, batch 1,
mean of 20 timed repeats after 5 warmups, dense measured in the same run.

```
=== 64 Q-heads / 4 KV-heads, GQA ratio 16 (paper Table 5 shape) ===

  seqlen=16384 (16k)   dense 8.443 ms
   threshold   sparsity   blasst ms   speedup
       1e-09      0.00%      10.027    0.842x
         1.0     21.57%       8.309    1.016x
         1.3     39.13%       7.768    1.087x
         1.7     49.98%       7.474    1.130x
         4.0     64.62%       7.087    1.191x
         6.0     78.19%       6.754    1.250x
         8.0     88.57%       6.572    1.285x
        10.0     93.88%       6.518    1.295x

  seqlen=65536 (64k)   dense 136.845 ms
   threshold   sparsity   blasst ms   speedup
       1e-09      0.00%     160.776    0.851x
         0.7     20.88%     129.989    1.053x
         0.8     33.95%     122.974    1.113x
         0.9     46.07%     117.185    1.168x
         1.1     63.57%     110.111    1.243x
         2.0     80.96%     104.839    1.305x
         6.0     89.40%     101.844    1.344x
        10.0     96.09%     101.065    1.354x

=== 32 Q-heads / 8 KV-heads, GQA ratio 4 (Qwen3-8B shape) ===

  seqlen=16384 (16k)   dense 4.251 ms
   threshold   sparsity   blasst ms   speedup
       1e-09      0.00%       5.127    0.829x
         1.0     21.50%       4.263    0.997x
         1.3     39.14%       3.975    1.070x
         1.7     50.17%       3.831    1.110x
         4.0     64.58%       3.647    1.166x
         6.0     77.89%       3.482    1.221x
         8.0     88.43%       3.409    1.247x
        10.0     93.82%       3.384    1.256x

  seqlen=65536 (64k)   dense 69.721 ms
   threshold   sparsity   blasst ms   speedup
       1e-09      0.00%      80.017    0.871x
         0.7     20.80%      65.361    1.067x
         0.8     33.89%      62.067    1.123x
         0.9     46.05%      59.414    1.173x
         1.1     63.60%      56.012    1.245x
         2.0     81.00%      53.086    1.313x
         6.0     89.33%      51.587    1.352x
        10.0     96.02%      50.676    1.376x
```

Thresholds span roughly 0% to 95% sparsity at each length, the same range as
Table 5 of the BLASST paper, rather than clustering around 50% as an earlier
version of this table did. The sparsity column is populated at every length,
including 64k: `sparsity_reference.py` batches the head and query-tile axes
into GPU ops instead of looping over them in Python, which brings a 64k count
down to a few seconds and makes the `--sparsity-all` / `--max-sparsity-seqlen`
gate from the earlier version unnecessary. These sparsity figures match the
kernel's own (since-removed) atomic counters to the digit at every threshold
tested against them, including this range.

Against the same sweep run outside the container (ROCm 7.0 / torch 2.10 instead
of 7.2 / 2.11), all 32 rows agree to within 0.006x, so the numbers are not
sensitive to the exact toolchain pins.

## Stage 2: RULER per-layer speed

`scripts/ruler_speed.py`, Qwen3-8B, per-layer attention time summed over 36
layers, dense measured in the same run.

Full run, all 13 tasks, ctx=32768, threshold 0.03, mean over tasks:

```
  configuration                                speedup
  dense                                         1.000x
  skip-softmax only                             0.980x
  + defer_v_load                                0.942x
  + dynamic work counter                        1.133x
```

The middle two rows are the reason the work counter ships with the feature and
not after it. Skipping the matmul does not pay for the check on its own, and
deferring V makes it worse, because both are averaged away by whichever head in
the layer is least sparse. Those two rows predate the counter and cannot be
reproduced from this bundle, because a nonzero threshold now switches the
counter on unconditionally. They are quoted from the earlier measurement.

The last row is what this bundle produces, and is also what `ruler_speed.py`
now reports directly as its single `speedup` column (commit `2dfd88e4` dropped
the separate skip-only timing, since it was never actually isolating the
scheduler from the threshold check -- see the note above). Per task, against
the same measurement taken outside the container:

```
Task                  bundle   earlier     diff
cwe                   1.090x    1.090x   +0.000
fwe                   1.202x    1.202x   +0.000
niah_multikey_1       1.151x    1.151x   +0.000
niah_multikey_2       1.059x    1.060x   -0.001
niah_multikey_3       1.094x    1.094x   +0.000
niah_multiquery       1.152x    1.153x   -0.001
niah_multivalue       1.114x    1.114x   +0.000
niah_single_1         1.130x    1.131x   -0.001
niah_single_2         1.114x    1.116x   -0.002
niah_single_3         1.114x    1.115x   -0.001
qa_1                  1.143x    1.143x   +0.000
qa_2                  1.247x    1.246x   +0.001
vt                    1.112x    1.113x   -0.001
MEAN                  1.133x    1.133x   +0.000
```

Every task agrees to within 0.002x, well inside run-to-run noise.

The sparsity column is now populated at this length too (mean 55.2%, replay
sparsity): the vectorized counter in `sparsity_reference.py` makes an
O(seqlen^2) count over 32768 tokens cheap enough to run unconditionally, so
`--no-sparsity` is no longer needed here.

Note the aggregation. A task's speedup is its layer times summed and then
divided, so each layer counts for as much time as it takes. Averaging the
per-layer ratios instead gives the cheap early layers, which have little
sparsity to exploit, equal weight with the expensive ones, and reads about
0.01x higher. Both are in the JSON. Quoting the two interchangeably is an easy
way to produce a spurious disagreement.

`QUICK=1`, 2 tasks, ctx=8192, threshold 0.03, as a smoke test. Speedups are
lower than at 32768 because attention is a smaller share of the work at 8k, so
do not compare these against the table above:

```
Task                  sparsity    speedup   >=1.0x
niah_single_1            44.8%     1.058x   30/36
qa_1                     42.6%     1.059x   29/36
MEAN                     43.7%     1.059x
```

Note the sparsity column here is replay sparsity and reads high. See the
caveats in README.md.

## Stage 3: RULER accuracy

`scripts/ruler_accuracy.py`, Qwen3-8B, ctx=32768, threshold 0.03, 13 tasks x 50
samples (650 prompts per arm), reproduced through this bundle:

```
Task                         Dense   BLASST     Drop
------------------------- -------- -------- --------
cwe                          83.0%    78.6%    -4.4%
fwe                          94.0%    88.7%    -5.3%
niah_multikey_1              98.0%    96.0%    -2.0%
niah_multikey_2             100.0%    98.0%    -2.0%
niah_multikey_3             100.0%    96.0%    -4.0%
niah_multiquery              99.5%    98.5%    -1.0%
niah_multivalue              98.0%    98.5%    +0.5%
niah_single_1               100.0%   100.0%    +0.0%
niah_single_2               100.0%   100.0%    +0.0%
niah_single_3               100.0%   100.0%    +0.0%
qa_1                          66.0%    66.0%    +0.0%
qa_2                          52.0%    56.0%    +4.0%
vt                           100.0%    97.6%    -2.4%
------------------------- -------- -------- --------
AGGREGATE                   91.58%   90.30%   -1.28%
```

Ran in about 2h37m end to end (dense arm 4752s, BLASST arm 4720s). Matches the
earlier pre-bundle measurement's -1.28 pt figure exactly, per task as well as
in aggregate.

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
