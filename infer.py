"""Synthesise one text with the engine and write a 24 kHz wav.

    python infer.py --text "Your order will arrive tomorrow." --out out.wav                      # stored bench voice
    python infer.py --text "..." --ref-wav my_voice.wav --ref-text "what the clip says" --out out.wav
    python infer.py --model 1.7B --lang German --text "Guten Morgen." --seed 7 --out de.wav

This is the same code path as gen_engine.py (Q3Engine: add_request / step / pop_events), for one request.
Without --ref-wav the voice is samples/voice_ref_<model>.pt (written by gen_stock.py).  With --ref-wav the voice-clone
prompt (speaker vector + reference codes + transcript) is built with the upstream wrapper first, then installed into the
engine exactly as Q3Engine.__init__ installs the stored voice (prompt parts + primed decoder snapshot).
The first run compiles and captures graphs (minutes); --fast-boot skips the decoder compile (same audio, slower decode).
"""
import argparse, os, sys, time, types
import numpy as np, soundfile as sf, torch

ROOT = os.environ.get("QWEN3TTS_ROOT") or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--text", required=True); ap.add_argument("--out", default="out.wav")
ap.add_argument("--model", default="0.6B", choices=["0.6B", "1.7B"])
ap.add_argument("--lang", default="English", help="English, Chinese, German, Spanish, French, Italian, Portuguese, Russian, Japanese, Korean")
ap.add_argument("--ref-wav", default=None); ap.add_argument("--ref-text", default=None)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-frames", type=int, default=1500, help="frame cap (12.5 frames per second)")
ap.add_argument("--fast-boot", action="store_true", help="do not torch.compile the decoder (boot ~1 min instead of several)")
a = ap.parse_args()

voice = None
if a.ref_wav:
    if not a.ref_text: sys.exit("--ref-text (the transcript of --ref-wav) is required in voice-clone mode")
    from qwen_tts import Qwen3TTSModel
    up = Qwen3TTSModel.from_pretrained(f"{ROOT}/weights/Qwen3-TTS-12Hz-{a.model}-Base", device_map="cuda:0", dtype=torch.bfloat16)
    wav, sr = sf.read(a.ref_wav, dtype="float32")
    if wav.ndim > 1: wav = wav.mean(1)
    it = up.create_voice_clone_prompt(ref_audio=(wav, sr), ref_text=a.ref_text)[0]
    voice = dict(ref_code=it.ref_code.cpu(), ref_spk_embedding=it.ref_spk_embedding.float().cpu(), ref_text=it.ref_text)
    del up; torch.cuda.empty_cache()

from engine.engine import Q3Engine

extra = f"seed={a.seed},keep_audio=1,cap={a.max_frames},admit=8" + (",dec_compile=off" if a.fast_boot else "")
if a.ref_wav: extra += ",fast_admit=0"            # graphed admission caches the prompt prefix of the stored voice at boot; a custom voice installed afterwards must use the original prefill path
t0 = time.time(); eng = Q3Engine(model=a.model, n_seats=64, extra=extra); print(f"[infer] engine ready in {time.time() - t0:.0f} s")
if voice is not None:                                    # same three steps as Q3Engine.__init__ does for the stored voice
    eng.voice = eng.pb.voice(types.SimpleNamespace(**voice))
    eng.dec.prime(0, voice["ref_code"]); eng.snap = eng.dec.snapshot(0); eng.dec.reset([0])
    torch.cuda.synchronize()

t0 = time.perf_counter(); rid = eng.add_request(a.text, a.lang); first = None; done = False
while not done:
    if not eng.step(): time.sleep(0.001)
    for r, kind, t, n in eng.pop_events():
        if r != rid: continue
        if kind == "audio" and first is None: first = t
        elif kind == "done": done = True
wall = time.perf_counter() - t0
pcm = np.concatenate(eng.audio[rid]); sf.write(a.out, pcm, eng.sample_rate)
sec = len(pcm) / eng.sample_rate
print(f"[infer] {a.out}: {sec:.2f} s of audio, first audio after {(first - t0) * 1e3:.0f} ms, total {wall * 1e3:.0f} ms (real-time factor {wall / sec:.3f})")
eng.close()
