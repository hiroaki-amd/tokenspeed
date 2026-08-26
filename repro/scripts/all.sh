#!/usr/bin/env bash
# Run the full reproduction inside the container. See README.md.
#
#   bash scripts/all.sh            # everything, several hours
#   QUICK=1 bash scripts/all.sh    # smoke test, roughly 20 minutes
#
# Results land in results/ as both a log and a JSON file per stage.

set -euo pipefail

OUT="${OUT:-results}"
mkdir -p "$OUT"

CTX="${CTX:-32768}"
THRESHOLD="${THRESHOLD:-0.03}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"

# QUICK writes to *_quick.* rather than to the same names the full run uses.
# Without the suffix a smoke test silently overwrites the full run's results,
# which is a bad trade: the smoke test takes twenty minutes and what it
# clobbers took several hours. The suffix also matches how the committed
# reference logs are named, so a QUICK run diffs against the right file.
SUFFIX=""
if [[ -n "${QUICK:-}" ]]; then
    echo "QUICK mode: 2 tasks, 3 samples each, ctx 8192."
    CTX=8192
    TASK_ARGS=(--tasks niah_single_1,qa_1)
    ACC_ARGS=(--num-samples 3)
    SUFFIX="_quick"
else
    TASK_ARGS=()
    ACC_ARGS=()
fi

echo
echo "############ 0/3  reference self-check ############"
echo "Tests the sparsity counter against the kernel before anything quotes it."
# Deliberately not fatal, and note the `|| status=$?` is also what keeps
# `set -e` from killing the run here. A failure invalidates the sparsity
# columns of stages 1 and 2, but speed and accuracy are measured off the kernel
# and never go through the reference, so aborting would discard hours of valid
# measurement over a counter bug. Record it and repeat it at the end, where it
# cannot be scrolled past.
verify_status=0
python scripts/verify_reference.py 2>&1 | tee "$OUT/verify_reference.log" \
    || verify_status=$?
if [[ $verify_status -ne 0 ]]; then
    echo
    echo "!!!! reference self-check FAILED. The sparsity columns below are not"
    echo "!!!! trustworthy. Speed and accuracy are unaffected. Continuing."
fi

echo
echo "############ 1/3  synthetic sparsity sweep ############"
# No SUFFIX here, nor on the self-check above: QUICK does not change either
# stage, so both produce the full result and overwriting is what you want.
python scripts/sparsity_sweep.py \
    --output-file "$OUT/sparsity_sweep.json" \
    2>&1 | tee "$OUT/sparsity_sweep.log"

echo
echo "############ 2/3  RULER per-layer speed ############"
python scripts/ruler_speed.py \
    --model "$MODEL" \
    --context-length "$CTX" \
    --threshold "$THRESHOLD" \
    "${TASK_ARGS[@]}" \
    --output-file "$OUT/ruler_speed$SUFFIX.json" \
    2>&1 | tee "$OUT/ruler_speed$SUFFIX.log"

echo
echo "############ 3/3  RULER accuracy ############"
echo "This is the long one. Skip it with SKIP_ACCURACY=1."
if [[ -z "${SKIP_ACCURACY:-}" ]]; then
    python scripts/ruler_accuracy.py \
        --model "$MODEL" \
        --context-length "$CTX" \
        --threshold "$THRESHOLD" \
        "${TASK_ARGS[@]}" "${ACC_ARGS[@]}" \
        --output-file "$OUT/ruler_accuracy$SUFFIX.json" \
        2>&1 | tee "$OUT/ruler_accuracy$SUFFIX.log"
else
    echo "skipped."
fi

echo
echo "done. results in $OUT/"
if [[ $verify_status -ne 0 ]]; then
    echo
    echo "!!!! Reminder: the reference self-check failed at the start of this"
    echo "!!!! run. Do not quote any sparsity figure above. See"
    echo "!!!! $OUT/verify_reference.log."
    exit 1
fi
