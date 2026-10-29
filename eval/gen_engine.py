"""The 32 gate sentences through OUR engine, all at once (voice-clone ICL, upstream sampling settings).
Usage: gen_engine.py <0.6B|1.7B> <seed> [label] [extra]   -> samples/engine_<model>_s<seed>/<key>.wav + meta.json"""
import sys, os, time, json, types, numpy as np, soundfile as sf
ROOT = os.environ.get("QWEN3TTS_ROOT") or os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from eval.eval_texts_q3 import ALL, LANG_NAME
from engine.engine import Q3Engine
M, seed = sys.argv[1], int(sys.argv[2]); label = sys.argv[3] if len(sys.argv) > 3 else f"engine_{M}_s{seed}"
extra = f"seed={seed},keep_audio=1,cap=400,admit=8," + (sys.argv[4] if len(sys.argv) > 4 else "dec_compile=off")
out = f"{ROOT}/samples/{label}"; os.makedirs(out, exist_ok=True)
eng = Q3Engine(model=M, n_seats=64, extra=extra)
t0 = time.perf_counter(); rids = {}; first = {}; done = {}
for lang, key, text in ALL: rids[eng.add_request(text, LANG_NAME[lang])] = key
while len(done) < len(rids):
    if not eng.step(): time.sleep(0.001)
    for rid, kind, t, n in eng.pop_events():
        if kind == "audio": first.setdefault(rid, t)
        else: done[rid] = t
wall = time.perf_counter() - t0; meta = {}
for rid, key in rids.items():
    w = np.concatenate(eng.audio[rid]); sf.write(f"{out}/{key}.wav", w, 24000)
    meta[key] = {"seconds": len(w) / 24000, "frames": round(len(w) / 1920), "ttfa_ms": (first[rid] - t0) * 1e3}
json.dump({"model": M, "seed": seed, "wall_s": wall, "extra": extra, "clips": meta, "stats": eng.stats()}, open(f"{out}/meta.json", "w"), indent=1)
fr = [v["frames"] for v in meta.values()]
print(f"GEN engine {M} seed {seed}: {len(meta)} clips, {sum(v['seconds'] for v in meta.values()):.1f} s audio, mean {sum(fr)/len(fr):.1f} frames (min {min(fr)} max {max(fr)}), wall {wall:.1f} s, truncated text {eng.stats()['text_truncated']}")
