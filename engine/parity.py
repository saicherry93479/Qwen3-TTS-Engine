"""Teacher-forced parity against upstream on a real generation, plus a free-running sanity clip.
MODEL=0.6B|1.7B  COMPILE=off|default"""
import os, sys, time, json, torch, numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.eval_texts_q3 import EN
from engine.model_load import load_upstream, extract, ROOT
from engine.talker import Talker
from engine.prompt import PromptBuilder
import torch.nn.functional as F
M = os.environ.get("MODEL", "0.6B"); COMPILE = os.environ.get("COMPILE", "off"); NF = int(os.environ.get("FRAMES", "48"))
DT = torch.float32 if os.environ.get("DTYPE", "bf16") == "fp32" else torch.bfloat16
m = load_upstream(M, dtype=DT); core = m.model; talker = core.talker; cp = talker.code_predictor
ref_wav, sr = sf.read(f"{ROOT}/samples/voice_ref.wav", dtype="float32")
item = m.create_voice_clone_prompt(ref_audio=(ref_wav, sr), ref_text=EN[0])[0]
text = EN[1]
# ---- record upstream
rec = {"t_in": [], "t_logits": [], "t_codes": [], "cp_logits": []}
t_fwd, cp_fwd = talker.forward, cp.forward
import functools
@functools.wraps(t_fwd)
def t_hook(*a, **k):
    o = t_fwd(*a, **k)
    if k.get("inputs_embeds") is not None and k["inputs_embeds"].shape[1] > 1: rec["t_in"].append(k["inputs_embeds"].detach().clone())
    rec["t_logits"].append(o.logits[:, -1].float().clone())
    if o.hidden_states[1] is not None: rec["t_codes"].append(o.hidden_states[1].clone())
    return o
@functools.wraps(cp_fwd)
def cp_hook(*a, **k):
    o = cp_fwd(*a, **k); rec["cp_logits"].append(o.logits[:, -1].float().clone()); return o
talker.forward, cp.forward = t_hook, cp_hook
torch.manual_seed(1234)
wavs, osr = m.generate_voice_clone(text=text, language="English", voice_clone_prompt=[item], max_new_tokens=NF)
talker.forward, cp.forward = t_fwd, cp_fwd
codes_up = torch.cat(rec["t_codes"], 0)                       # [frames,16]; frame f recorded at generate call f
n = codes_up.shape[0]; print(f"[upstream] {n} frames, {len(rec['t_logits'])} talker calls, {len(rec['cp_logits'])} predictor calls", flush=True)
# ---- ours
W = extract(m); pb = PromptBuilder(m); voice = pb.voice(item)
pre, trail, pad = pb.build(voice, text, "English")
up_in = rec["t_in"][0][0]
print(f"[prompt] ours {tuple(pre.shape)} upstream {tuple(up_in.shape)} identical={torch.equal(pre, up_in) if pre.shape == up_in.shape else False} trailing {tuple(trail.shape)}", flush=True)
T = Talker(W, n_seats=8, seat_len=512, buckets=[1, 4], compile_mode=COMPILE, cp_compile=COMPILE)
rel = lambda a, b: float((a - b).norm() / b.norm())
h = T.prefill_hidden([0], [pre]); T.pos[0] = pre.shape[0]
st = torch.tensor([0], device=W.device)
tr, cr, tam, cam = [], [], [], []
for f in range(n):
    lg = F.linear(h, W.codec_head).float(); up = rec["t_logits"][f]
    tr.append(rel(lg, up)); tam.append(int(lg.argmax(-1)[0] == up.argmax(-1)[0]))
    c = codes_up[f:f + 1]
    cl = T.cp.tf_logits(h, c)[0]; upc = torch.cat(rec["cp_logits"][15 * f:15 * f + 15], 0)
    cr.append([rel(cl[i], upc[i]) for i in range(15)]); cam.append([int(cl[i].argmax() == upc[i].argmax()) for i in range(15)])
    embs = [W.codec_emb[c[:, 0]]] + [W.cp_emb[i][c[:, i + 1]] for i in range(15)]
    x = torch.stack(embs, 1).sum(1) + (trail[f] if f < trail.shape[0] else pad)
    h = (T.decode if COMPILE != "off" else T._decode)(x.to(W.dtype), W.cos[T.pos[st]][:, None, :], W.sin[T.pos[st]][:, None, :], st * T.seat_len + T.pos[st], (T.pos[st] + 1).to(torch.int32), T.seat_pages[st]); T.pos[st] += 1
tr, cr = np.array(tr), np.array(cr)
res = dict(model=M, compile=COMPILE, frames=n, prompt_identical=bool(pre.shape == up_in.shape and torch.equal(pre, up_in)),
           talker_rel_mean=float(tr.mean()), talker_rel_max=float(tr.max()), talker_argmax_agree=float(np.mean(tam)),
           cp_rel_mean=float(cr.mean()), cp_rel_max=float(cr.max()), cp_argmax_agree=float(np.mean(cam)))
print("[parity]", json.dumps(res), flush=True)
print("[parity] cp rel by step:", " ".join(f"{v:.3f}" for v in cr.mean(0)), "| argmax by step:", " ".join(f"{v:.2f}" for v in np.array(cam).mean(0)), flush=True)
print("[parity] prompt max abs diff", float((pre.float()-up_in.float()).abs().max()), "rel", rel(pre.float(), up_in.float()), "| talker rel first/last", tr[0], tr[-1], flush=True)
res["dtype"] = str(DT)[6:]
json.dump(res, open(f"{ROOT}/results/parity_{M}_{COMPILE}_{res['dtype']}.json", "w"), indent=1)
if DT == torch.float32: print("PARITY_DONE"); sys.exit(0)
# ---- free-running clip through the frame graph (4 streams, bucket 4)
T.pos.zero_(); T.capture([4])
texts = [EN[1], EN[2], EN[3], EN[5]]; built = [pb.build(voice, t, "English") for t in texts]
first = T.admit([0, 1, 2, 3], [b[0] for b in built], [b[1] for b in built], pad)
frames = [first.clone()]; t0 = time.time()
for _ in range(200):
    frames.append(T.step([0, 1, 2, 3]).clone())
torch.cuda.synchronize(); print(f"[free-run] 200 frames x 4 streams in {time.time()-t0:.2f} s ({(time.time()-t0)*5:.1f} ms/frame)", flush=True)
codes = torch.stack(frames, 1)                                 # [4, F, 16]
for i in range(4):
    c = codes[i]; e = (c[:, 0] == W.eos).nonzero()
    L = int(e[0]) if len(e) else c.shape[0]
    full = torch.cat([voice["ref_code"], c[:L]], 0)
    wav, fs = core.speech_tokenizer.decode([{"audio_codes": full}])
    cut = int(voice["ref_code"].shape[0] / full.shape[0] * wav[0].shape[0])
    sf.write(f"{ROOT}/samples/engine_{M}_{COMPILE}_{i}.wav", wav[0][cut:], fs)
    print(f"[free-run] stream {i}: {L} frames ({L*0.08:.1f} s){'' if len(e) else ' NO EOS'}", flush=True)
print("PARITY_DONE")
