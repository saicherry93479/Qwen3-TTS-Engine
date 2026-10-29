#!/bin/bash
# Quality gate: the 32 gate sentences through OUR engine for one seed, scored against the upstream ("stock") clips.
# Stock clips for seeds 1000 and 2000 are generated first if they are missing (they define the seed-to-seed spread).
#   scripts/run_gate.sh <0.6B|1.7B> <seed>   -> samples/engine_<model>_s<seed>/, results/gate_engine_<model>_s<seed>.json
# (The overnight runs of 2026-10-05 wrote results/gate_engine_<model>.json for the original arrival path and
#  results/gate_engine_<model>_fa.json for graphed admission, which is now the engine default.)
ROOT="${QWEN3TTS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; cd "$ROOT" || exit 1
M="${1:?model: 0.6B or 1.7B}"; S="${2:-1000}"; PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python
mkdir -p results logs samples
for s in 1000 2000; do
  [ -f "samples/stock_${M}_s$s/meta.json" ] || "$PY" eval/gen_stock.py "$M" $s > "logs/gen_stock_${M}_s$s.log" 2>&1
done
"$PY" eval/gen_engine.py "$M" "$S" > "logs/gen_engine_${M}_s$S.log" 2>&1; grep -E "^GEN|rror" "logs/gen_engine_${M}_s$S.log"
"$PY" eval/score.py "stock_${M}_s1000" "stock_${M}_s2000" "engine_${M}_s$S" --out "results/gate_engine_${M}_s$S.json" 2>&1 | grep -vE "Warning|warn" | tail -12
