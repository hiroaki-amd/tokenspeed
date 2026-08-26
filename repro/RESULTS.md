# Reference numbers

What this bundle produced on MI350X, for comparison against your own run. All
of it was measured through `docker/run.sh`, against kernel commit `7d742874`
("Skip the softmax update only when the whole query tile votes to skip").

All three stages below are from that kernel, including the two `QUICK=1` smoke
tests, which are marked where they appear.

Every sparsity figure below has been recounted since `e58dcecf` fixed
`scripts/sparsity_reference.py`, which until then advanced the running max per
row rather than per unanimous block (the pre-`7d742874` rule). Because the
running max feeds the next block's decision, that error compounded along the
sequence and undercounted. Stage 1 moved a long way as a result and the table
below is the recount. Stages 2 and 3 did not move at all, for a reason worth
knowing: the two rules can only disagree on a block whose vote was not
unanimous, and at threshold 0.03 the disagreement never grows large enough to
change a later block's decision. Running both recurrences side by side on the
same captured RULER activations gave identical counts on every one of 36 layers
for each of three tasks, so stage 2's numbers are unchanged rather than
unrecounted. Speed and accuracy never go through the reference at all.

The raw logs and the per-layer JSON behind every table below are committed in
`results/`; see `results/README.md` for what each file is and which ones not to
quote.

## Stage 1: synthetic sparsity sweep

`scripts/sparsity_sweep.py`, random Q/K/V, causal, head_dim=128, bf16, batch 1,
mean of 20 timed repeats after 5 warmups, dense measured in the same run.

```
=== 64 Q-heads / 4 KV-heads, GQA ratio 16 (paper Table 5 shape) ===

  seqlen=16384 (16k)   dense 8.442 ms
   threshold   sparsity   blasst ms   speedup
       1e-09      0.00%       9.387    0.899x
         1.0     21.57%       7.519    1.123x
         1.3     44.30%       6.677    1.264x
         1.7     62.56%       6.089    1.387x
         4.0     90.20%       5.393    1.565x
         6.0     94.80%       5.314    1.589x
         8.0     96.62%       5.285    1.597x
        10.0     97.54%       5.278    1.600x

  seqlen=65536 (64k)   dense 136.718 ms
   threshold   sparsity   blasst ms   speedup
       1e-09      0.00%     150.417    0.909x
         0.7     20.88%     117.741    1.161x
         0.8     33.95%     109.361    1.250x
         0.9     46.07%     102.346    1.336x
         1.1     64.16%      93.050    1.469x
         2.0     87.36%      84.069    1.626x
         6.0     97.86%      83.241    1.642x
        10.0     99.04%      83.398    1.639x

=== 32 Q-heads / 8 KV-heads, GQA ratio 4 (Qwen3-8B shape) ===

  seqlen=16384 (16k)   dense 4.238 ms
   threshold   sparsity   blasst ms   speedup
       1e-09      0.00%       4.850    0.874x
         1.0     21.50%       3.883    1.091x
         1.3     44.35%       3.425    1.237x
         1.7     62.76%       3.129    1.355x
         4.0     90.26%       2.820    1.503x
         6.0     94.84%       2.784    1.522x
         8.0     96.62%       2.776    1.526x
        10.0     97.53%       2.779    1.525x

  seqlen=65536 (64k)   dense 69.499 ms
   threshold   sparsity   blasst ms   speedup
       1e-09      0.00%      75.064    0.926x
         0.7     20.80%      59.239    1.173x
         0.8     33.89%      55.198    1.259x
         0.9     46.05%      51.960    1.338x
         1.1     64.20%      47.464    1.464x
         2.0     87.38%      42.631    1.630x
         6.0     97.86%      40.824    1.702x
        10.0     99.04%      40.682    1.708x
```

Against the same sweep on the pre-`7d742874` kernel, every one of the 32 rows
is faster, and the gain widens with sparsity: at 64k on the Qwen3-8B shape,
threshold 10.0 went 1.376x to 1.708x. The dense column is unchanged to within
0.3%, which is what makes the two comparable. The floor moved too: at threshold
1e-09, where nothing is skipped and only the cost of checking remains, the
overhead eased from 0.842x-0.871x to 0.874x-0.926x, because the branch that
zeroed dissenting rows out of `p` is gone.

Thresholds span roughly 0% to 99% sparsity at each length, covering the range
of Table 5 of the BLASST paper, rather than clustering around 50% as an earlier
version of this table did. The sparsity column is populated at every length,
including 64k: `sparsity_reference.py` batches the head and query-tile axes
into GPU ops instead of looping over them in Python, which brings a 64k count
down to a few seconds and makes the `--sparsity-all` / `--max-sparsity-seqlen`
gate from the earlier version unnecessary.

This table is the recount on the fixed counter, and it moved a long way from
the pre-`e58dcecf` figures: at 16k threshold 4.0 read 64.6% before and reads
90.2% now, at 64k threshold 2.0 read 81.0% and reads 87.4%. The low thresholds
(1e-09 through 1.0, and 0.7 through 0.9 at 64k) are unchanged, which is the
expected shape: the two rules can only diverge once dissenting blocks are
common enough for a suppressed running max to flip a later decision. Timings
were remeasured in the same run and agree with the earlier ones to within
0.5%.

The 0.006x agreement this sweep previously showed against a run outside the
container (ROCm 7.0 / torch 2.10 instead of 7.2 / 2.11) was measured on the old
kernel and has not been repeated on this one. Nothing suggests the toolchain
sensitivity changed, but it is no longer a claim this file can back.

## Stage 2: RULER per-layer speed

`scripts/ruler_speed.py`, Qwen3-8B, per-layer attention time summed over 36
layers, dense measured in the same run.

This is the measurement that matters. Full run, all 13 tasks, ctx=32768,
threshold 0.03, with the sparsity each task's replayed activations produced:

```
Task                  sparsity    speedup   >=1.0x
cwe                      49.8%     1.227x   34/36
fwe                      57.5%     1.368x   36/36
niah_multikey_1          54.9%     1.305x   34/36
niah_multikey_2          45.3%     1.181x   32/36
niah_multikey_3          53.6%     1.235x   32/36
niah_multiquery          54.8%     1.305x   34/36
niah_multivalue          55.0%     1.262x   33/36
niah_single_1            61.0%     1.286x   35/36
niah_single_2            55.0%     1.261x   33/36
niah_single_3            54.9%     1.260x   33/36
qa_1                     58.1%     1.303x   33/36
qa_2                     60.4%     1.424x   35/36
vt                       57.8%     1.258x   34/36
MEAN                     55.2%     1.283x
```

Against the pre-`7d742874` kernel, which read 1.133x on this same table, every
one of the 13 tasks improved:

```
Task                 before    after     diff
cwe                  1.090x   1.227x   +0.136
fwe                  1.202x   1.368x   +0.166
niah_multikey_1      1.151x   1.305x   +0.154
niah_multikey_2      1.059x   1.181x   +0.122
niah_multikey_3      1.094x   1.235x   +0.141
niah_multiquery      1.152x   1.305x   +0.153
niah_multivalue      1.114x   1.262x   +0.148
niah_single_1        1.130x   1.286x   +0.156
niah_single_2        1.114x   1.261x   +0.147
niah_single_3        1.114x   1.260x   +0.146
qa_1                 1.143x   1.303x   +0.159
qa_2                 1.247x   1.424x   +0.177
vt                   1.112x   1.258x   +0.146
MEAN                 1.133x   1.283x   +0.150
```

Layers at or above 1.0x went from 410 of 468 to 438. The dense arm is what
makes this a fair comparison: summed over all 13 tasks it came to 7826.7 ms
before and 7827.5 ms now, a difference of 0.01%, so the two runs saw the same
machine in the same state and only the BLASST arm moved.

The gain comes from what the old rule cost, not from skipping more blocks per
se. Under it a block with even one dissenting row still had the voting rows
zeroed out of `p`, which held those rows' running max down and made them less
likely to clear the threshold on every later block. Removing that recovers both
speed and accuracy at a fixed threshold.

The sparsity column above did not move when the counter was fixed, unlike
stage 1's. That is worth a sentence, because "unchanged" is also what a still
broken counter would produce. Running the old per-row recurrence and the fixed
per-block one side by side over the same captured activations, at this
threshold, gives identical block counts on all 36 layers of every task tried
(`cwe` 49.84%, `qa_2` 60.40%, `niah_multikey_2` 45.32%, each identical under
both rules). It is not that votes are rare: 36% to 52% of blocks here are voted
but not unanimously. It is that at 0.03 the max a dissenting block would have
held back is never far enough below the true running max to flip a later
block's decision. Rerunning the whole stage with the fixed counter reproduced
every task's sparsity exactly and every speedup to within 0.002x.

It is still replay sparsity, though, and reads higher than a live run
(README.md).

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
niah_single_1            44.8%     1.157x   34/36
qa_1                     42.6%     1.161x   30/36
MEAN                     43.7%     1.159x
```

On the old kernel this read 1.059x, with 30 and 29 layers at or above 1.0x.
Sparsity is unchanged to the decimal shown, the same agreement between the two
counting rules that stage 2 shows at this threshold. Two consecutive runs of
this smoke test came out 1.159x and 1.173x, so read the third decimal as noise:
at 8k the per-layer times are small enough that run-to-run variance is visible,
which is another reason not to lean on this table.

Note the sparsity column here is replay sparsity and reads high. See the
caveats in README.md.

## Stage 3: RULER accuracy

`scripts/ruler_accuracy.py`, Qwen3-8B, ctx=32768, threshold 0.03, 13 tasks x 50
samples (650 prompts per arm), reproduced through this bundle:

```
Task                         Dense   BLASST     Drop
------------------------- -------- -------- --------
cwe                          83.0%    82.4%    -0.6%
fwe                          94.0%    92.0%    -2.0%
niah_multikey_1              98.0%    98.0%    +0.0%
niah_multikey_2             100.0%   100.0%    +0.0%
niah_multikey_3             100.0%   100.0%    +0.0%
niah_multiquery              99.5%    99.5%    +0.0%
niah_multivalue              98.0%    98.5%    +0.5%
niah_single_1               100.0%   100.0%    +0.0%
niah_single_2               100.0%   100.0%    +0.0%
niah_single_3               100.0%   100.0%    +0.0%
qa_1                         66.0%    68.0%    +2.0%
qa_2                         52.0%    54.0%    +2.0%
vt                          100.0%   100.0%    +0.0%
------------------------- -------- -------- --------
AGGREGATE                   91.58%   91.72%   +0.15%
```

Ran in about 2h36m end to end (dense arm 4761s, BLASST arm 4616s).

The dense arm is bit-identical to what the old kernel's dense arm produced,
task for task, which is the check that the two runs are comparable: dense is
the same code path either way, and threshold 0.03 is the only thing that
differs between the arms.

**Only 2 of 13 tasks degrade, and the aggregate does not.** On the old per-row
rule the same threshold cost 1.28 points; here it gains 0.15. The two tasks
that still move are the same two that moved most before, and both improved:
`fwe` -5.3 -> -2.0 and `cwe` -4.4 -> -0.6. Everything else is flat or better,
and the three tasks that read as gains (`niah_multivalue` +0.5, `qa_1` +2.0,
`qa_2` +2.0) are within what 50 samples per task can resolve; they are not
evidence that approximating attention helps.

The reason is the same one behind the speedup. The old rule zeroed a
dissenting block's voting rows out of `p`, which both discarded their
contribution and held their running max down, compounding into later blocks.
Discarding the vote instead leaves those rows bit-identical to dense, so the
only rows that lose anything are the ones in a block the whole tile agreed to
skip.

`QUICK=1` is 3 samples per task at ctx 8192 and is far too small to say
anything about accuracy. It exists to prove the pipeline runs. For the record
it came out at 100.0% on `niah_single_1` and 33.3% on `qa_1`, identical on both
arms.

## Sparsity

The replay figure at threshold 0.03 and ctx 32768 is 55.2%, from the stage 2
table above, and the fixed counter reproduces it exactly (see that section for
why the fix did not move it). The 54.9% this section used to give was the same
quantity on the old kernel. The live figure it paired that with, 43.9%, was
measured on the old kernel and does not carry over; the live rate on the
current kernel is around 60%, measured outside this bundle, and nothing in this
bundle produces a live number.

The distinction the section exists to make still holds: `ruler_speed.py`
captures activations under dense attention and replays them, so its sparsity
reads higher than a real run, where the approximation perturbs each layer's
input. Timings are unaffected, since they depend only on the tensors passed in.
See README.md.
