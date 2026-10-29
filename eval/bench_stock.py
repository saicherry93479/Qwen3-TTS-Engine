"""Stock Qwen3-TTS (upstream code) on the A100: where does the time go per frame, at batch 1..64.
3-layer split: talker step (28 layers), code predictor (15 autoregressive steps of a 5-layer model, nested HF generate),
codec decode.  MODEL=0.6B|1.7B"""
import os, sys, time, json, torch, soundfile as sf
ROOT = os.environ.get("QWEN3TTS_ROOT") or os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from eval.eval_texts_q3 import EN
from qwen_tts import Qwen3TTSModel
M = os.environ.get("MODEL", "0.6B"); BATCHES = [int(x) for x in os.environ.get("BATCHES", "1,4,16,64").split(",")]
FRAMES = int(os.environ.get("FRAMES", "125"))            # 125 frames = 10 s at 12.5 Hz
m = Qwen3TTSModel.from_pretrained(f"{ROOT}/weights/Qwen3-TTS-12Hz-{M}-Base", device_map="cuda:0", dtype=torch.bfloat16)
core = m.model; talker = core.talker; cp = talker.code_predictor
n = lambda mod: sum(p.numel() for p in mod.parameters()) / 1e6
print(f"[params] talker(with cp) {n(talker):.0f}M  talker.model {n(talker.model):.0f}M  code_predictor {n(cp):.0f}M  speaker_enc {n(core.speaker_encoder) if getattr(core,'speaker_encoder',None) is not None else 0:.1f}M  codec {n(core.speech_tokenizer.model) if hasattr(core.speech_tokenizer,'model') else -1:.0f}M", flush=True)
T = {}
def timed(obj, name, key):
    fn = getattr(obj, name)
    def wrap(*a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter(); r = fn(*a, **k); torch.cuda.synchronize()
        T[key] = T.get(key, 0.0) + time.perf_counter() - t0; T[key + "_n"] = T.get(key + "_n", 0) + 1; return r
    setattr(obj, name, wrap)
timed(talker.model, "forward", "talker"); timed(cp, "generate", "cp")
st = core.speech_tokenizer
for nm in ("decode",):
    if hasattr(st, nm): timed(st, nm, "dec")
ref_wav, sr = sf.read(f"{ROOT}/samples/voice_ref.wav", dtype="float32")
prompt = m.create_voice_clone_prompt(ref_audio=(ref_wav, sr), ref_text=EN[0])
text = EN[1] + " " + EN[2]
rows = []
for bs in BATCHES:
    kw = dict(text=[text] * bs, language=["English"] * bs, voice_clone_prompt=prompt, max_new_tokens=FRAMES)
    for rep in range(3):
        T.clear(); torch.cuda.synchronize(); t0 = time.perf_counter()
        wavs, osr = m.generate_voice_clone(**kw)
        torch.cuda.synchronize(); wall = time.perf_counter() - t0
    audio = sum(len(w) for w in wavs) / osr; steps = T.get("cp_n", 0)
    row = dict(model=M, bs=bs, wall_ms=wall * 1e3, frames=steps, audio_s=audio, talker_ms=T.get("talker", 0) * 1e3, cp_ms=T.get("cp", 0) * 1e3,
               dec_ms=T.get("dec", 0) * 1e3, per_frame_ms=wall * 1e3 / max(steps, 1), rtf=wall / audio, xrt=audio / wall, vram_gb=torch.cuda.max_memory_allocated() / 2**30)
    row["talker_only_ms"] = row["talker_ms"]; row["other_ms"] = row["wall_ms"] - row["talker_ms"] - row["cp_ms"] - row["dec_ms"]
    rows.append(row)
    print("[stock {model}] bs {bs:3d} | wall {wall_ms:8.0f} ms for {frames} frames ({per_frame_ms:6.1f} ms/frame) | talker {talker_ms:7.0f} | code predictor {cp_ms:7.0f} | decode {dec_ms:6.0f} | other {other_ms:7.0f} | audio {audio_s:6.1f} s | RTF {rtf:.3f} | {xrt:5.1f}x realtime | {vram_gb:.1f} GB".format(**row), flush=True)
json.dump(rows, open(f"{ROOT}/results/stock_{M}.json", "w"), indent=1)
sf.write(f"{ROOT}/samples/stock_{M}.wav", wavs[0], osr)
print("STOCK_DONE")
