# Reference logs

The raw output behind `../RESULTS.md`, committed so the write-up can be checked
against what the scripts actually printed. Your own runs land in this directory
too and are gitignored.

```
sparsity_sweep.log / .json           stage 1, both GQA shapes, 16k and 64k
ruler_speed_ctx32768.log / .json     stage 2, all 13 tasks, ctx 32768, th 0.03
ruler_speed_quick.log / .json        stage 2 under QUICK=1, 2 tasks, ctx 8192
ruler_accuracy_quick.log / .json     stage 3 under QUICK=1, 3 samples per task
```

The JSON carries per-layer detail that the log summarises away: for every layer
of every task, dense / skip-only / V-deferred times, both speedups, and sparsity
when it was measured.

Three things to know before reading these.

`ruler_speed_quick` predates the aggregation fix in commit `99de33bb`, so its
speedups are means of the per-layer ratios and read about 0.01x higher than the
same data aggregated the way `ruler_speed_ctx32768` and `RESULTS.md` do. It is
kept as the smoke-test record, not as a number to quote.

`ruler_accuracy_quick` is 3 samples per task. That is enough to show the
pipeline runs end to end and nothing else. The accuracy claim in `RESULTS.md`
rests on the 650-prompt run, which is not reproduced here.

`ruler_speed_ctx32768` was run with `--no-sparsity`, so its sparsity column is
`n/a`. The reference counter is a Python loop over blocks and is `O(seqlen^2)`,
which is impractical at 32768 across 13 tasks. Sparsity does not depend on the
toolchain, so nothing is lost.
