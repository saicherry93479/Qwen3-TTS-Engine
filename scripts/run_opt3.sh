#!/bin/bash
# OPT-3 A/B: GEMM row-split plan + sort top-k (Q3_GEMM_PLAN=1 Q3_SORT_TOPK=1) on top of dec_fine=1, 90 s windows.
cd /home/charan/qwen3tts_bench
run() { env $6 flock gpu.lock .venv/bin/python engine/bench_open.py --model $2 --seats $3 --max-live $3 --streams-list $4 --duration ${DUR:-90} --warmup ${WARM:-15} \
     --extra "$5,req_log=logs/req_$1.jsonl" --json-out results/$1.json > logs/$1.log 2>&1
  echo "== $1 ($5 $6): recompile warnings $(grep -c recompile_limit logs/$1.log)"; grep -E "^\[load\]|^\[engine\]|rror" logs/$1.log | cut -c1-260; }
run opt3_0.6B_plan 0.6B 640 576,640 dec_fine=1 "Q3_GEMM_PLAN=1 Q3_SORT_TOPK=1"
run opt3_1.7B_plan 1.7B 544 544,576 dec_fine=1 "Q3_GEMM_PLAN=1 Q3_SORT_TOPK=1"
run opt3_1.7B_base 1.7B 544 544,576 dec_fine=1 "Q3_GEMM_PLAN=0"
echo OPT3_DONE
