#!/bin/bash
# Download the Qwen3-TTS checkpoints used here into weights/ (never commit them). ~7.4 GB.
#   scripts/download_weights.sh [0.6B|1.7B|all]
ROOT="${QWEN3TTS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; W="${1:-all}"
HF="$ROOT/.venv/bin/hf"; [ -x "$HF" ] || HF=hf
mkdir -p "$ROOT/weights"
get() { HF_HUB_ENABLE_HF_TRANSFER=1 "$HF" download "Qwen/$1" --local-dir "$ROOT/weights/$1" && echo "ok $1 $(du -sh "$ROOT/weights/$1" | cut -f1)"; }
[ "$W" = "0.6B" ] || [ "$W" = "all" ] && get Qwen3-TTS-12Hz-0.6B-Base
[ "$W" = "1.7B" ] || [ "$W" = "all" ] && get Qwen3-TTS-12Hz-1.7B-Base
get Qwen3-TTS-Tokenizer-12Hz
