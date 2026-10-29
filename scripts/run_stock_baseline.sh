#!/bin/bash
# Timing of the UPSTREAM Qwen3-TTS code (nested generate): ms per frame split into talker / code predictor / decode,
# at batch 1, 4, 16, 64.  Measured on the A100: 178-194 ms per 80 ms frame at any batch size.
#   scripts/run_stock_baseline.sh <0.6B|1.7B> [batches]   -> results/stock_<model>.json, samples/stock_<model>.wav
ROOT="${QWEN3TTS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; cd "$ROOT" || exit 1
M="${1:-0.6B}"; PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python
mkdir -p results logs samples
MODEL="$M" BATCHES="${2:-1,4,16,64}" "$PY" eval/bench_stock.py > "logs/stock_$M.log" 2>&1
grep -E "^\[stock|^\[params|STOCK_DONE|rror" "logs/stock_$M.log" | cut -c1-240
