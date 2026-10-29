"""Qwen3-TTS serving loop: talker + code predictor on fixed seats (one frame graph per batch bucket), arrivals admitted
by a bounded prefill, exact streaming codec decoder batched over seats with a chunk ramp.

Implements the interface at the top of engine/bench_open.py (make_engine / add_request / step / pop_events / live /
stats / close / sample_rate).

One step():
  1. admit up to `admit` pending requests: build the voice-clone prompt, prefill all prompt tokens but the last (eager,
     packed), restore the decoder state of the voice (reference codes already decoded), and leave the LAST prompt token as
     the seat's pending talker input, so the first frame is produced by the same bulk graph as every other frame
     (same maths as upstream's prefill-then-sample; no eager first-frame path).
  2. bulk frame graph for all live seats (talker decode -> codebook 0 -> 15 code-predictor steps -> next input), codes
     written to a per-seat GPU buffer, codebook 0 copied to pinned memory (the only host read of the step).
  3. decoder: seats whose pending frames reach their next chunk size (ramp, e.g. 1,2,4) are decoded in one batched call
     per chunk size on a side stream. First packets are decoded and awaited inside the step (time to first audio);
     steady chunks are launched and collected one step later, so they overlap the next language-model step.
Seats are freed only when their last PCM has reached the host ("done").

Options (bench_open --extra "k=v,k=v"): ramp=1/2/4  admit=8  seat_len=768  dec_dtype=fp16  dec_max=256  seed=0
  dec_compile=default|off  cap=400 (frame cap)  serial=1 (sync after each stage: exact GPU split, no overlap)  keep_audio=1
"""
import os, sys, time, json, threading, types
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.model_load import load_upstream, extract
from engine.talker import Talker
from engine.prompt import PromptBuilder
from engine.decoder_stream import StreamDecoder, HOP
from engine.arrivals import FastAdmit

FRAME_S = 0.08


def _opts(extra):
    o = {}
    for kv in (extra or "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1); o[k.strip()] = v.strip()
    return o


class Q3Engine:
    sample_rate = 24000

    def __init__(self, model="0.6B", n_seats=512, max_live=None, extra=""):
        o = _opts(extra); t0 = time.time()
        self.model = model; self.n_seats = n_seats; self.max_live = max_live or n_seats
        self.ramp = [int(x) for x in o.get("ramp", "1/2/4").split("/")]
        self.admit_max = int(o.get("admit", "8")); self.seat_len = int(o.get("seat_len", "768"))
        self.serial = o.get("serial", "0") == "1"; self.keep_audio = o.get("keep_audio", "0") == "1"
        self.dec_max = int(o.get("dec_max", "256")); seed = int(o.get("seed", "0"))
        self.frame_budget = o.get("frame_budget", os.environ.get("Q3_FRAME_BUDGET", "flat"))   # 'text' = 48 + 0.90 x (chars + 7 x digits) frames per request (research/runaway_generation.md); default flat cap
        self.rinfo = {}; self.reqlog = open(o["req_log"], "a", buffering=1) if o.get("req_log") else None
        self.cap = int(o.get("cap", "100000"))                               # frame cap for normal requests (EOS still ends them)
        dec_dt = dict(fp16=torch.float16, fp32=torch.float32)[o.get("dec_dtype", "fp16")]
        self.dev = torch.device("cuda:0")
        m = load_upstream(model); self.m = m
        if getattr(m.model, "speech_tokenizer", None) is not None:
            del m.model.speech_tokenizer; torch.cuda.empty_cache()            # we decode with our own streaming decoder
        self.W = extract(m, max_pos=max(4096, self.seat_len)); self.pb = PromptBuilder(m)
        v = torch.load(f"{ROOT}/samples/voice_ref_{model}.pt", weights_only=False)
        self.voice = self.pb.voice(types.SimpleNamespace(ref_text=v["ref_text"], ref_code=v["ref_code"], ref_spk_embedding=v["ref_spk_embedding"]))
        buckets = sorted(set([1, 2, 4, 8] + list(range(16, n_seats + 1, 16)) + [n_seats]))
        self.T = Talker(self.W, n_seats=n_seats, seat_len=self.seat_len, buckets=buckets, max_text=int(o.get("max_text", "640")), seed=seed)
        self.T.capture()
        # graphed admission with per-(voice, language) prefix reuse (engine/arrivals.py); fast_admit=0 -> eager prefill
        self.fast = o.get("fast_admit", os.environ.get("Q3_FAST_ADMIT", "1")) == "1"
        self.first_lane = self.fast and o.get("first_lane", os.environ.get("Q3_FIRST_LANE", "0")) == "1"
        self.fl_rounds = int(o.get("fl_rounds", "4"))
        if self.fast:
            self.fa = FastAdmit(self.T, self.pb, m, compile_mode="default")
            self.slot = {l: self.fa.voice_slot(self.voice, l) for l in ("English", "Spanish", "German")}
            self.fa.capture()
            if self.first_lane: self.fa.enable_first_frame(seed=seed + 7919)
        t1 = time.time()
        # ---- decoder: slot == seat; voice state primed once and copied into a seat at admission
        self.dec = StreamDecoder(n_slots=n_seats, dtype=dec_dt)
        self.chunks = sorted(set(self.ramp) | {1})
        steady = self.ramp[-1]; self.dec_buckets = {}
        for n in self.chunks:
            if o.get("dec_fine", os.environ.get("Q3_DEC_FINE", "0")) == "1":   # fine buckets: 32 % of decoded frame-rows were padding
                bs = list(range(16, 161, 16)) + [192, 256, 384, 512] if n == steady else [1, 2, 4, 8, 32, 128]   # every 8 rows ran out of memory at 704 seats (each graph keeps its activations)
            else:
                bs = [16, 32, 64, 96, 128, 192, 256, 384, 512] if n == steady else [8, 32, 128]
            self.dec_buckets[n] = [b for b in bs if b <= max(self.dec_max, 8)]
        dc = o.get("dec_compile", "default")
        for n in self.chunks:
            self.dec.capture(buckets=tuple(self.dec_buckets[n]), chunks=(n,), compile=(dc != "off"), graphs=True, mode=dc if dc != "off" else "default")
            torch.cuda.empty_cache()
        self.dec.prime(0, v["ref_code"]); self.snap = self.dec.snapshot(0); self.dec.reset([0])
        torch.cuda.synchronize()
        # ---- per-seat host state
        S1 = n_seats + 1
        self.frames = np.zeros(S1, np.int64); self.maxf = np.zeros(S1, np.int64); self.emitted = np.zeros(S1, np.int64)
        self.nchunks = np.zeros(S1, np.int64); self.fin = np.zeros(S1, bool); self.fixed = np.zeros(S1, bool)
        self.inflight = np.zeros(S1, np.int64); self.rid = np.full(S1, -1, np.int64)
        self.free = list(range(n_seats - 1, -1, -1)); self.act = np.zeros(0, np.int64); self.drain = set()
        self.codes = torch.zeros(S1, 16, self.seat_len, dtype=torch.int16, device=self.dev)
        self.ar16 = torch.arange(16, device=self.dev)[None, :]
        self.ar_n = {n: torch.arange(n, device=self.dev)[None, :] for n in self.chunks}
        self.h_c0 = torch.zeros(n_seats, dtype=torch.int64).pin_memory()
        self.side = torch.cuda.Stream()
        # pinned staging for decoder launches (ring; a slot is reused only after its launch was collected)
        self.ring = [dict(idx=torch.zeros(512, dtype=torch.int64).pin_memory(), start=torch.zeros(512, dtype=torch.int64).pin_memory(),
                          pcm={n: torch.zeros(max(self.dec_buckets[n]), n * HOP, dtype=torch.float32).pin_memory() for n in self.chunks}, busy=False) for _ in range(24)]
        self.pend_dec = []                                   # launches not yet collected
        self.lock = threading.Lock(); self.pending = []; self.next_rid = 0; self.events = []; self.audio = {}
        self.eos = int(self.W.eos)
        self.st = dict(steps=0, rejected=0, admitted=0, finished=0, gpu_n=0, gpu_talker_ms=0.0, gpu_cp_ms=0.0, gpu_dec_ms=0.0, gpu_prefill_ms=0.0,
                       gpu_lm_ms=0.0, dec_calls=0, dec_rows=0, dec_pad_rows=0, lm_rows=0, lm_pad_rows=0, text_truncated=0, host_admit_ms=0.0, wait_first_ms=0.0)
        self.ratio = self._load_ratio()
        self._warm()
        print(f"[engine] {model}: {n_seats} seats x {self.seat_len}, talker buckets {len(buckets)}, decoder chunks {self.chunks} ramp {self.ramp} "
              f"buckets { {n: len(b) for n, b in self.dec_buckets.items()} }, boot LM {t1 - t0:.0f} s + decoder {time.time() - t1:.0f} s, "
              f"vram {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)

    # ------------------------------------------------------------------ helpers
    def _load_ratio(self):
        """talker share of the language-model step by live count, from the step bench (the frame graph is one unit)."""
        try:
            rows = json.load(open(f"{ROOT}/results/step_{self.model}.json"))["rows"]
            return sorted((r["B"], r["talker_ms"] / (r["talker_ms"] + r["cp_ms"])) for r in rows)
        except Exception:
            return [(1, 0.25), (1024, 0.3)]

    def _talker_share(self, B):
        xs = [b for b, _ in self.ratio]; ys = [y for _, y in self.ratio]
        return float(np.interp(B, xs, ys))

    @torch.inference_mode()
    def _warm(self):
        """Run the admission path once per admit size so nothing compiles or allocates lazily under load."""
        if self.fast:
            prep = self.fa.prepare("This is a warm up sentence for the engine.", "English", self.slot["English"])
            for n in range(1, self.admit_max + 1): self.fa.admit(list(range(n)), [prep] * n)
            self.T.pos[:self.admit_max] = 0; torch.cuda.synchronize(); return
        pre, trail, pad = self.pb.build(self.voice, "This is a warm up sentence for the engine.", "English")
        for n in range(1, self.admit_max + 1):
            seats = list(range(n)); self._prefill(seats, [pre] * n, [trail] * n, pad)
        self.T.pos[:self.admit_max] = 0; torch.cuda.synchronize()

    def _prefill(self, seats, embeds, trailing, pad):
        T = self.T; st = torch.as_tensor(seats, device=self.dev)
        T.seen[st] = False; T.ngen[st] = 0
        for s, tr in zip(seats, trailing):
            n = int(tr.shape[0])
            if n > T.max_text: self.st["text_truncated"] += 1; n = T.max_text
            if n: T.trail[s, :n] = tr[:n].to(T.dt)
            T.trail[s, n] = pad.to(T.dt); T.trail_len[s] = n
        T.prefill_hidden(seats, [e[:-1] for e in embeds])                      # KV of every prompt token but the last
        T.pos[st] = torch.tensor([int(e.shape[0]) - 1 for e in embeds], device=self.dev)
        T.x_in[st] = torch.stack([e[-1] for e in embeds], 0).to(T.dt)          # the last prompt token is decoded by the frame graph

    # ------------------------------------------------------------------ interface
    def add_request(self, text, lang, voice=None, seed=None, max_frames=None):
        with self.lock:
            if len(self.pending) + self.n_live >= self.max_live:
                self.st["rejected"] += 1; return False
            raw = text if isinstance(text, str) else ""; nch = len(raw); ndig = sum(c.isdigit() for c in raw)
            if self.fast: text = self.fa.prepare(text, lang, self.slot[lang])   # tokeniser only, on the caller thread
            self.next_rid += 1; rid = self.next_rid
            ntok = (int(text.n_trail) + int((text.rem != self.fa.pad_id).sum())) if self.fast else -1
            self.rinfo[rid] = (nch, ndig, ntok, int(48 + 0.90 * (nch + 7 * ndig)))
            self.pending.append((rid, text, lang, max_frames)); return rid

    @property
    def n_live(self): return len(self.act) + len(self.drain)

    def live(self): return self.n_live
    def stats(self): return dict(self.st)
    def pop_events(self): ev, self.events = self.events, []; return ev

    def close(self): pass

    # ------------------------------------------------------------------ decoder launches
    def _launch(self, seats, n, stream=None):
        """One batched decode of chunk size n for `seats` (np array) on the side stream. Returns launch records."""
        out = []; mb = max(self.dec_buckets[n]); stream = stream or self.side
        for a in range(0, len(seats), mb):
            piece = seats[a:a + mb]; B = len(piece)
            slot = next(r for r in self.ring if not r["busy"]); slot["busy"] = True
            slot["idx"][:B] = torch.from_numpy(piece); slot["start"][:B] = torch.from_numpy(self.emitted[piece])
            with torch.cuda.stream(stream):
                e0 = torch.cuda.Event(enable_timing=True); e0.record()
                idx = slot["idx"][:B].to(self.dev, non_blocking=True); start = slot["start"][:B].to(self.dev, non_blocking=True)
                codes = self.codes[idx[:, None], :, start[:, None] + self.ar_n[n]]             # [B, n, 16]
                pcm = self.dec.decode(idx, codes.permute(0, 2, 1).long())
                slot["pcm"][n][:B].copy_(pcm, non_blocking=True)
                e1 = torch.cuda.Event(enable_timing=True); e1.record()
            bk = next(b for b in self.dec_buckets[n] if b >= B)
            self.st["dec_calls"] += 1; self.st["dec_rows"] += B * n; self.st["dec_pad_rows"] += (bk - B) * n
            self.emitted[piece] += n; self.nchunks[piece] += 1; self.inflight[piece] += 1
            out.append((e0, e1, piece, n, slot))
        return out

    def _collect(self, launches):
        """Wait for the launches, emit their audio events (PCM is on the host now) and finish drained seats."""
        for e0, e1, piece, n, slot in launches:
            e1.synchronize(); now = time.perf_counter()
            self.st["gpu_dec_ms"] += e0.elapsed_time(e1)
            pcm = slot["pcm"][n]
            for i, s in enumerate(piece):
                rid = int(self.rid[s]); self.events.append((rid, "audio", now, n * HOP))
                if self.keep_audio: self.audio.setdefault(rid, []).append(pcm[i].numpy().copy())
            self.inflight[piece] -= 1; slot["busy"] = False
            for s in piece:
                if self.fin[s] and self.inflight[s] == 0 and self.frames[s] == self.emitted[s]: self._finish(int(s), now)

    def _budget(self, rid):
        return self.rinfo[rid][3] if self.frame_budget == "text" else 10**9

    def _finish(self, s, now):
        info = self.rinfo.pop(int(self.rid[s]), None)            # (was defined twice: the second definition shadowed this one and req_log stayed empty)
        if self.reqlog is not None and info is not None and not self.fixed[s]:
            self.reqlog.write(json.dumps(dict(frames=int(self.frames[s]), chars=info[0], digits=info[1], text_tokens=info[2], budget=info[3], capped=bool(self.frames[s] >= self.maxf[s]))) + "\n")
        self.events.append((int(self.rid[s]), "done", now, 0)); self.drain.discard(s); self.rid[s] = -1
        self.free.append(s); self.st["finished"] += 1

    # ------------------------------------------------------------------ one step
    # ------------------------------------------------------------------ fast lane (first_lane=1)
    def _fast_lane(self, new):
        """Prefill + frame 0 + first 1-frame decode for the new requests on the high-priority stream, while the bulk frame
        graph is running on the default stream. Returns the seats; they join the bulk lane from frame 1 (next step)."""
        th = time.perf_counter(); hp = self.fa.hp
        one = [l for l in self.pend_dec if l[3] == 1]            # a 1-frame tail decode in flight uses the same decoder graphs
        if one: self._collect(one); self.pend_dec = [l for l in self.pend_dec if l[3] != 1]
        seats, preps = [], []; cap = self.seat_len - self.fa.L - 2
        for rid, prep, lang, mf in new:
            self.st["text_truncated"] += int(prep.truncated)
            s = self.free.pop()
            self.frames[s] = 1; self.emitted[s] = 0; self.nchunks[s] = 0; self.fin[s] = False; self.inflight[s] = 0; self.rid[s] = rid
            self.fixed[s] = mf is not None; self.maxf[s] = min(mf, cap) if mf is not None else min(cap, self.cap, self._budget(rid))
            seats.append(s); preps.append(prep)
        with torch.cuda.stream(hp):
            p0 = torch.cuda.Event(enable_timing=True); p0.record()
            self.dec.restore(seats, self.snap)
        c0 = self.fa.admit_fast(seats, preps)                    # async on hp: prefill + frame 0 -> codes [n,16]
        with torch.cuda.stream(hp):
            st = torch.as_tensor(seats, device=self.dev)
            self.codes[st, :, 0] = c0.to(torch.int16)
            p1 = torch.cuda.Event(enable_timing=True); p1.record()
        L = self._launch(np.asarray(seats, np.int64), 1, stream=hp)
        self._collect(L)                                         # first PCM on the host: "audio" events
        self.st["gpu_prefill_ms"] += p0.elapsed_time(p1); self.st["admitted"] += len(seats); self.st["fl_calls"] = self.st.get("fl_calls", 0) + 1
        self.st["host_admit_ms"] += (time.perf_counter() - th) * 1e3
        return seats

    @torch.inference_mode()
    def _step_fl(self):
        T = self.T; act = self.act; B = len(act)
        with self.lock: has_new = bool(self.pending) and bool(self.free)
        if not has_new and B == 0 and not self.drain and not self.pend_dec: return False
        e1 = None
        if B:
            e0 = torch.cuda.Event(enable_timing=True); e0.record()
            T.pos[T.n_seats] = 0
            o = T.step(act)                                                    # bulk frame graph (async, default stream)
            seat_t = T.s_seat[:B]
            self.codes[seat_t[:, None], self.ar16, (T.ngen[seat_t] - 1)[:, None]] = o.to(torch.int16)
            self.h_c0[:B].copy_(o[:, 0], non_blocking=True)
            e1 = torch.cuda.Event(enable_timing=True); e1.record()
        joined = []; rounds = 0
        while True:                                                            # arrivals are served while the bulk graph runs
            with self.lock:
                k = min(len(self.pending), self.admit_max, len(self.free)); new, self.pending = self.pending[:k], self.pending[k:]
            if new:
                joined += self._fast_lane(new); rounds += 1
                if rounds >= self.fl_rounds or (e1 is not None and e1.query()): break
            elif e1 is None or e1.query(): break
            else: time.sleep(0.0002)
        if B:
            e1.synchronize()
            lm = e0.elapsed_time(e1); sh = self._talker_share(B)
            self.st["gpu_lm_ms"] += lm; self.st["gpu_talker_ms"] += lm * sh; self.st["gpu_cp_ms"] += lm * (1 - sh)
            bk = next(b for b in T.buckets if b >= B); self.st["lm_rows"] += B; self.st["lm_pad_rows"] += bk - B
            c0 = self.h_c0[:B].numpy()
            eos = (c0 == self.eos) & ~self.fixed[act]
            self.frames[act] += ~eos
            fin = eos | (self.frames[act] >= self.maxf[act])
            self.st["ended_by_cap"] = self.st.get("ended_by_cap", 0) + int((~eos & (self.frames[act] >= self.maxf[act]) & ~self.fixed[act]).sum())
            self.st["max_frames_seen"] = max(self.st.get("max_frames_seen", 0), int(self.frames[act].max()))
            if fin.any():
                done = act[fin]; self.fin[done] = True; self.drain.update(int(s) for s in done); act = act[~fin]
        if joined: act = np.concatenate([act, np.asarray(joined, np.int64)])
        self.act = act
        return self._decode_sched()

    @torch.inference_mode()
    def step(self):
        if self.first_lane: return self._step_fl()
        T = self.T
        with self.lock:
            k = min(len(self.pending), self.admit_max, len(self.free)); new, self.pending = self.pending[:k], self.pending[k:]
        if not new and len(self.act) == 0 and not self.drain and not self.pend_dec: return False
        p0 = p1 = None
        if new:
            th = time.perf_counter(); p0 = torch.cuda.Event(enable_timing=True); p0.record()
            seats, embeds, trails = [], [], []
            for rid, text, lang, mf in new:
                if self.fast: pre, trail, L = text, None, self.fa.L; self.st["text_truncated"] += int(text.truncated)
                else: pre, trail, pad = self.pb.build(self.voice, text, lang); L = int(pre.shape[0])
                s = self.free.pop(); cap = self.seat_len - L - 2
                self.frames[s] = 0; self.emitted[s] = 0; self.nchunks[s] = 0; self.fin[s] = False; self.inflight[s] = 0; self.rid[s] = rid
                self.fixed[s] = mf is not None; self.maxf[s] = min(mf, cap) if mf is not None else min(cap, self.cap, self._budget(rid))
                seats.append(s); embeds.append(pre); trails.append(trail)
            if self.fast: self.fa.admit(seats, embeds)
            else: self._prefill(seats, embeds, trails, pad)
            self.dec.restore(seats, self.snap)
            p1 = torch.cuda.Event(enable_timing=True); p1.record()
            self.act = np.concatenate([self.act, np.asarray(seats, np.int64)]); self.st["admitted"] += len(seats)
            self.st["host_admit_ms"] += (time.perf_counter() - th) * 1e3
        act = self.act; B = len(act)
        if B:
            e0 = torch.cuda.Event(enable_timing=True); e0.record()
            T.pos[T.n_seats] = 0                                               # scratch seat (padded rows) advances every replay: keep it inside the seat
            o = T.step(act)                                                    # frame graph (async)
            seat_t = T.s_seat[:B]
            self.codes[seat_t[:, None], self.ar16, (T.ngen[seat_t] - 1)[:, None]] = o.to(torch.int16)
            self.h_c0[:B].copy_(o[:, 0], non_blocking=True)
            e1 = torch.cuda.Event(enable_timing=True); e1.record()
            e1.synchronize()
            lm = e0.elapsed_time(e1); sh = self._talker_share(B)
            self.st["gpu_lm_ms"] += lm; self.st["gpu_talker_ms"] += lm * sh; self.st["gpu_cp_ms"] += lm * (1 - sh)
            bk = next(b for b in T.buckets if b >= B); self.st["lm_rows"] += B; self.st["lm_pad_rows"] += bk - B
            if p0 is not None: self.st["gpu_prefill_ms"] += p0.elapsed_time(p1)
            c0 = self.h_c0[:B].numpy()
            eos = (c0 == self.eos) & ~self.fixed[act]
            self.frames[act] += ~eos                                           # the EOS frame is not audio
            fin = eos | (self.frames[act] >= self.maxf[act])
            self.st["ended_by_cap"] = self.st.get("ended_by_cap", 0) + int((~eos & (self.frames[act] >= self.maxf[act]) & ~self.fixed[act]).sum())
            self.st["max_frames_seen"] = max(self.st.get("max_frames_seen", 0), int(self.frames[act].max()))
            if fin.any():
                done = act[fin]; self.fin[done] = True; self.drain.update(int(s) for s in done); self.act = act[~fin]
        return self._decode_sched()

    def _decode_sched(self):
        # steady chunks launched in the previous step overlapped this step's language model: collect them now
        if self.pend_dec: self._collect(self.pend_dec); self.pend_dec = []
        # decoder scheduling
        cand = np.concatenate([self.act, np.fromiter(self.drain, np.int64, len(self.drain))]) if self.drain else self.act
        if len(cand):
            pend = self.frames[cand] - self.emitted[cand]
            need = np.asarray(self.ramp)[np.minimum(self.nchunks[cand], len(self.ramp) - 1)]
            size = np.where(pend >= need, need, 0)
            tail = self.fin[cand] & (pend > 0) & (pend < need)
            if tail.any():
                for n in self.chunks: size = np.where(tail & (pend >= n), n, size)   # largest available chunk <= pending
            first = (self.nchunks[cand] == 0) & (size > 0)
            if first.any():                                                    # first packets: decode now, wait, emit
                tw = time.perf_counter(); L = []
                for n in np.unique(size[first]): L += self._launch(cand[first & (size == n)], int(n))
                self._collect(L); self.st["wait_first_ms"] += (time.perf_counter() - tw) * 1e3
            rest = ~first & (size > 0)
            if rest.any():
                for n in np.unique(size[rest]):
                    grp = cand[rest & (size == n)]
                    if len(grp) > self.dec_max:                               # bound the decoder work per step: most pending first
                        grp = grp[np.argsort(-(self.frames[grp] - self.emitted[grp]), kind="stable")[:self.dec_max]]
                    self.pend_dec += self._launch(grp, int(n))
                if self.serial: self._collect(self.pend_dec); self.pend_dec = []
            # streams that ended with nothing left to decode (e.g. EOS right after a chunk boundary)
            now = time.perf_counter()
            for s in list(self.drain):
                if self.inflight[s] == 0 and self.frames[s] == self.emitted[s]: self._finish(s, now)
        self.st["steps"] += 1; self.st["gpu_n"] += 1
        return True


def make_engine(a):
    return Q3Engine(model=a.model, n_seats=a.seats, max_live=a.max_live, extra=getattr(a, "extra", ""))
