#!/bin/bash
# OPT-3b: gate through GEMM plan + sort top-k (one seed, both models), then 300 s validation of 0.6B at 640 and 576.
cd /home/charan/qwen3tts_bench
export Q3_GEMM_PLAN=1 Q3_SORT_TOPK=1
for m in 0.6B 1.7B; do
  flock gpu.lock .venv/bin/python eval/gen_engine.py $m 1000 engine_${m}_opt3_s1000 dec_fine=1 > logs/gen_engine_${m}_opt3.log 2>&1
  flock gpu.lock .venv/bin/python eval/score.py stock_${m}_s1000 engine_${m}_opt3_s1000 --out results/gate_engine_${m}_opt3.json > logs/score_${m}_opt3.log 2>&1
  echo "== gate $m: $(grep -E 'engine_' logs/score_${m}_opt3.log | tail -1 | cut -c1-200)"
done
flock gpu.lock .venv/bin/python engine/bench_open.py --model 0.6B --seats 640 --max-live 640 --streams-list 576,640 --duration 300 --warmup 45 \
  --extra "dec_fine=1,req_log=logs/req_long4_0.6B.jsonl" --json-out results/long4_0.6B_opt3.json > logs/long4_0.6B_opt3.log 2>&1
grep -E "^\[load\]|rror" logs/long4_0.6B_opt3.log | cut -c1-260
echo OPT3B_DONE
