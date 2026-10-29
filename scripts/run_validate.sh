#!/bin/bash
# Validation at one offered load: 45 s warm-up + 300 s window (the numbers to quote).
#   scripts/run_validate.sh <0.6B|1.7B> <streams> [extra engine options]
# Validated: 0.6B at 400, 512, 576; 1.7B at 256, 448, 480 (704 / 640 seats, graphed admission = engine default).
# SEATS=... overrides the seat count.
#   -> results/long_<model>_<streams>.json, logs/long_<model>_<streams>.log
ROOT="${QWEN3TTS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; cd "$ROOT" || exit 1
M="${1:?model: 0.6B or 1.7B}"; N="${2:?offered streams}"; PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python
case "$M" in 0.6B) S=704 ;; 1.7B) S=640 ;; *) echo "model must be 0.6B or 1.7B"; exit 1 ;; esac
SEATS="${SEATS:-$S}"
mkdir -p results logs
"$PY" engine/bench_open.py --model "$M" --seats $SEATS --max-live $SEATS --streams-list "$N" \
  --duration 300 --warmup 45 --extra "${3:-}" --json-out "results/long_${M}_$N.json" > "logs/long_${M}_$N.log" 2>&1
grep -E "^\[load\]|^\[bench\]|rror" "logs/long_${M}_$N.log" | cut -c1-240
echo "recompile-limit warnings (must be 0): $(grep -c recompile_limit "logs/long_${M}_$N.log")"
