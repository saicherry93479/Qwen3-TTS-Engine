"""Reference clips with the UPSTREAM Qwen3-TTS code (voice-clone ICL mode, upstream sampling defaults), all 32 gate
sentences in one batch.  Usage: gen_stock.py <0.6B|1.7B> <seed> [out_label]   -> samples/stock_<model>_s<seed>/<key>.wav
Also writes samples/voice_ref_<model>.pt (ref_code, ref_spk_embedding, ref_text) on first use."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, os, time, json, torch, soundfile as sf
from eval.eval_texts_q3 import ALL, LANG_NAME, REF_WAV, REF_TEXT
from qwen_tts import Qwen3TTSModel
M, seed = sys.argv[1], int(sys.argv[2]); ROOT = os.environ.get("QWEN3TTS_ROOT") or os.path.dirname(os.path.abspath(__file__))
out = f"{ROOT}/samples/" + (sys.argv[3] if len(sys.argv) > 3 else f"stock_{M}_s{seed}"); os.makedirs(out, exist_ok=True)
m = Qwen3TTSModel.from_pretrained(f"{ROOT}/weights/Qwen3-TTS-12Hz-{M}-Base", device_map="cuda:0", dtype=torch.bfloat16)
wav, sr = sf.read(REF_WAV, dtype="float32")
items = m.create_voice_clone_prompt(ref_audio=(wav, sr), ref_text=REF_TEXT)
it = items[0]
torch.save({"ref_code": it.ref_code.cpu(), "ref_spk_embedding": it.ref_spk_embedding.float().cpu(), "ref_text": it.ref_text,
            "x_vector_only_mode": it.x_vector_only_mode, "icl_mode": it.icl_mode, "ref_wav": REF_WAV}, f"{ROOT}/samples/voice_ref_{M}.pt")
print(f"[voice] ref frames {tuple(it.ref_code.shape)} spk {tuple(it.ref_spk_embedding.shape)}", flush=True)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
t0 = time.perf_counter()
wavs, osr = m.generate_voice_clone(text=[t for _, _, t in ALL], language=[LANG_NAME[l] for l, _, _ in ALL], voice_clone_prompt=items, max_new_tokens=400)
wall = time.perf_counter() - t0
meta = {}
for (lang, key, text), w in zip(ALL, wavs):
    sf.write(f"{out}/{key}.wav", w, osr); meta[key] = {"seconds": len(w) / osr, "frames": round(len(w) / osr * 12.5)}
json.dump({"model": M, "seed": seed, "wall_s": wall, "clips": meta}, open(f"{out}/meta.json", "w"), indent=1)
fr = [v["frames"] for v in meta.values()]
print(f"GEN stock {M} seed {seed}: {len(wavs)} clips, {sum(v['seconds'] for v in meta.values()):.1f} s audio, mean {sum(fr)/len(fr):.1f} frames (min {min(fr)} max {max(fr)}), wall {wall:.1f} s")
