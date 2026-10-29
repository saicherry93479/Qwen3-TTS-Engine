#!/bin/bash
# Open-loop load sweep in ONE boot: steady-state "burst" points, then Poisson arrivals at each offered load
# (15 s warm-up + 90 s window per load, so slightly optimistic; use run_validate.sh for numbers to quote).
#   scripts/run_load.sh <0.6B|1.7B> [streams-list] [extra engine options]
# Defaults are the validated configuration: graphed admission (engine default), 704 seats (0.6B, 68.9 GB) / 640 seats
# (1.7B, 67.4 GB). On a smaller budget: SEATS=384 scripts/run_load.sh 0.6B 256,400
#   -> results/open_<model>.json, logs/open_<model>.log
ROOT="${QWEN3TTS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; cd "$ROOT" || exit 1
M="${1:?model: 0.6B or 1.7B}"; PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python
case "$M" in
  0.6B) S=704; L="${2:-400,512,576,640}";     B="256,400,512" ;;
  1.7B) S=640; L="${2:-256,448,480,512}";     B="256,384,448" ;;
  *) echo "model must be 0.6B or 1.7B"; exit 1 ;;
esac
SEATS="${SEATS:-$S}"
mkdir -p results logs
"$PY" engine/bench_open.py --model "$M" --seats $SEATS --max-live $SEATS --burst-list "$B" --streams-list "$L" \
  --duration 90 --warmup 15 --extra "${3:-}" --json-out "results/open_$M.json" > "logs/open_$M.log" 2>&1
grep -E "^\[burst\]|^\[load\]|^\[bench\]|rror" "logs/open_$M.log" | cut -c1-240
echo "recompile-limit warnings (must be 0): $(grep -c recompile_limit "logs/open_$M.log")"
