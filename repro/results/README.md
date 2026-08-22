# Reference logs

The raw output behind `../RESULTS.md`, committed so the write-up can be checked
against what the scripts actually printed. Your own runs land in this directory
too and are gitignored.

```
sparsity_sweep.log / .json           stage 1, both GQA shapes, 16k and 64k
ruler_speed_ctx32768.log / .json     stage 2, all 13 tasks, ctx 32768, th 0.03
ruler_speed_quick.log / .json        stage 2 under QUICK=1, 2 tasks, ctx 8192
ruler_accuracy.log / .json           stage 3, all 13 tasks, 650 prompts/arm
ruler_accuracy_quick.log / .json     stage 3 under QUICK=1, 3 samples per task
```

The JSON carries per-layer detail that the log summarises away: for every layer
of every task, dense and blasst (shipped configuration) times, the speedup, and
sparsity when it was measured.

Two things to know before reading these.

Both `ruler_speed_ctx32768` and `ruler_speed_quick` were regenerated after
commit `2dfd88e4` dropped the skip-only column: `ruler_speed.py` now only times
dense against the shipped configuration (threshold + `defer_v_load=True`, which
also switches on the dynamic work counter unconditionally), reported as a
single `speedup` column. Earlier versions of these two files had a
"skip-only" / "V-deferred" pair of columns from before that change; those are
gone now, not just relabelled, because the merged kernel API never let the two
be isolated from each other in the first place. `ruler_speed_ctx32768` no
longer needs `--no-sparsity` either: the vectorized counter in
`sparsity_reference.py` makes a 32768-length count cheap enough to run
unconditionally, so its sparsity column is populated (mean 55.2%, replay
sparsity -- see the caveat in `../README.md`).

`ruler_accuracy_quick` is 3 samples per task. That is enough to show the
pipeline runs end to end and nothing else. `ruler_accuracy` is the full
650-prompt run (13 tasks x 50 samples) that `RESULTS.md`'s accuracy claim
actually rests on.
