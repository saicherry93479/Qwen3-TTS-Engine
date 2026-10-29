"""Objective quality gate for Qwen3-TTS runs.  Whisper-large-v3 WER (and CER) per language with a normaliser that keeps
letters/marks/numbers, WavLM x-vector speaker similarity (a) to the clip of the same sentence in the reference run and
(b) to the voice-reference recording, duration ratio and level difference vs the reference run.
Usage: score.py <ref_label> <label> [<label> ...] [--out results/gate_x.json]     (a label is a directory samples/<label>/<key>.wav)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, re, os, unicodedata, json, numpy as np, torch, soundfile as sf, librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoFeatureExtractor, WavLMForXVector
from eval.eval_texts_q3 import ALL, REF_WAV
ROOT = os.environ.get("QWEN3TTS_ROOT") or os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]; outp = None
if "--out" in args: i = args.index("--out"); outp = args[i + 1]; args = args[:i] + args[i + 2:]
ref, labels = args[0], args[1:]
wproc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")       # no pipeline: it imports torchcodec, which is broken on this box
wmodel = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", torch_dtype=torch.float16).cuda().eval()
@torch.inference_mode()
def transcribe(x16, lang):
    feats = wproc(x16, sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
    ids = wmodel.generate(feats, language=lang, task="transcribe", max_new_tokens=220, do_sample=False, num_beams=1)
    return wproc.batch_decode(ids, skip_special_tokens=True)[0]
fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv"); sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").cuda().eval()
def norm(s):
    s = unicodedata.normalize("NFC", s.lower())
    s = "".join(ch if (unicodedata.category(ch)[0] in "LMN" or ch.isspace()) else " " for ch in s)   # keep letters, marks, numbers; punctuation -> space
    return re.sub(r"\s+", " ", s).strip()
def ed(a, b):
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, y in enumerate(b, 1): prev, dp[j] = dp[j], min(dp[j] + 1, dp[j-1] + 1, prev + (x != y))
    return dp[-1]
def load16(p):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    return librosa.resample(x, orig_sr=sr, target_sr=16000), x, sr
@torch.inference_mode()
def emb(x16):
    i = fe(x16, sampling_rate=16000, return_tensors="pt", padding=True); return torch.nn.functional.normalize(sv(**{k: v.cuda() for k, v in i.items()}).embeddings, dim=-1)[0]
voice_emb = emb(load16(REF_WAV)[0])
ref_emb, ref_stat, out = {}, {}, {}
for lab in [ref] + [l for l in labels if l != ref]:
    w = {}; c = {}; sims, vsims, durr, lvl, durs, bad, worst = [], [], [], [], [], 0, []
    for lang, key, text in ALL:
        p = f"{ROOT}/samples/{lab}/{key}.wav"
        try: x16, x, sr = load16(p)
        except Exception: bad += 1; continue
        hyp = transcribe(x16, lang); r, h = norm(text), norm(hyp)
        we = ed(r.split(), h.split()); ce = ed(list(r.replace(" ", "")), list(h.replace(" ", "")))
        w.setdefault(lang, [0, 0]); c.setdefault(lang, [0, 0])
        w[lang][0] += we; w[lang][1] += len(r.split()); c[lang][0] += ce; c[lang][1] += len(r.replace(" ", ""))
        if we: worst.append((we / max(1, len(r.split())), key, h))
        e = emb(x16); rms = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9); dur = len(x) / sr; durs.append(dur)
        vsims.append(float((e * voice_emb).sum()))
        if lab == ref: ref_emb[key] = e; ref_stat[key] = (dur, rms)
        elif key in ref_emb:
            sims.append(float((e * ref_emb[key]).sum())); durr.append(dur / ref_stat[key][0]); lvl.append(rms - ref_stat[key][1])
    res = {"missing": bad, "mean_seconds": float(np.mean(durs)), "mean_frames": float(np.mean(durs)) * 12.5,
           "sim_voice_mean": float(np.mean(vsims)), "sim_voice_min": float(np.min(vsims)),
           "worst": [{"wer": round(a, 2), "key": k, "hyp": h} for a, k, h in sorted(worst, reverse=True)[:4]]}
    for lang in w: res[f"wer_{lang}"] = 100 * w[lang][0] / max(1, w[lang][1]); res[f"cer_{lang}"] = 100 * c[lang][0] / max(1, c[lang][1])
    tw = sum(v[0] for v in w.values()), sum(v[1] for v in w.values()); res["wer_all"] = 100 * tw[0] / max(1, tw[1])
    if lab != ref: res.update(sim_ref_mean=float(np.mean(sims)), sim_ref_min=float(np.min(sims)), dur_ratio=float(np.mean(durr)),
                              dur_ratio_min=float(np.min(durr)), dur_ratio_max=float(np.max(durr)), level_db=float(np.mean(lvl)), level_db_absmax=float(np.max(np.abs(lvl))))
    out[lab] = res
    line = f"{lab:22s} WER en {res.get('wer_en', 0):5.1f}% es {res.get('wer_es', 0):5.1f}% de {res.get('wer_de', 0):5.1f}% | CER es {res.get('cer_es', 0):4.1f}% de {res.get('cer_de', 0):4.1f}% | SIM-voice {res['sim_voice_mean']:.3f} (min {res['sim_voice_min']:.3f}) | {res['mean_frames']:.1f} frames"
    if lab != ref: line += f" | SIM-ref {res['sim_ref_mean']:.3f} (min {res['sim_ref_min']:.3f}) dur x{res['dur_ratio']:.2f} [{res['dur_ratio_min']:.2f}-{res['dur_ratio_max']:.2f}] level {res['level_db']:+.1f} dB"
    print(line, flush=True)
if outp is None: outp = f"{ROOT}/results/gate_{ref}.json"
json.dump(out, open(outp, "w"), indent=1, ensure_ascii=False); print("SCORE_DONE", outp)
