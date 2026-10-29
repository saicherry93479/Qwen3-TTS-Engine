"""Fast admission for the talker (ARRIVALS agent, 2026-10-05). Same maths as the eager path; rounding-level differences.

Eager path (engine.py today): PromptBuilder.build on the engine thread (tokeniser + 2 projection GEMMs per request) and
Talker.prefill_hidden (packed flash-attn varlen over the whole 77-token prompt, ~400 kernel launches) = ~7.7 ms of
blocking host time and ~7.7 ms of GPU per arrival, inside the engine step.

This module:
  1. text table: text_projection(text_embedding(id)) for every token id, computed once (a lookup afterwards; the per-token
     result of a GEMM does not depend on the other rows, differences are bf16 rounding of a different kernel choice).
  2. shared prefix: in upstream's voice-clone ICL layout (non_streaming_mode=False) the prompt is
         [3 role tokens][think, think_bos, lang, think_eos, speaker, pad -> 6 positions][codec_bos + reference codes]
     with the text track of the last block = [reference text, target text, tts_eos, tts_pad...].  The first
     k = 3 + 6 + len(reference text tokens) positions are identical for every request of the same (voice, language), and
     attention is causal, so their K/V are computed ONCE per (voice, language) and copied into the seat.
  3. graphed prefill of the remaining R = L - 1 - k positions (fixed for a voice: L = 9 + 1 + reference frames) for
     S in {1,2,4,8} sequences: one CUDA graph per S (own memory pool, private capture stream), token ids as static inputs,
     per-seat state reset and trailing-text table written inside the graph.  The last prompt token is left as the seat's
     pending input (frame 0 comes out of the frame graph), exactly as engine.py does.
  4. prepare(text, lang): tokenisation only, CPU only, thread-safe for ONE caller thread -> do it in add_request.

Usage (engine thread):   fa = FastAdmit(T, pb, m); slot = fa.voice_slot(voice, "English"); fa.capture()
                         prep = fa.prepare(text, "English", slot)          # caller thread
                         fa.admit(seats, preps)                            # replaces Q3Engine._prefill(...)
Switches: Q3_PF_COMPILE=default|off (compile the layer function; default off = graph only).
"""
import os, time
import numpy as np
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from .ops import rms, rope, mlp, sample_topk
from .code_predictor import CodePredictor


class Prep:
    __slots__ = ("slot", "rem", "trail", "n_trail", "L", "truncated")


class FastAdmit:
    def __init__(self, T, pb, m, seq_buckets=(1, 2, 4, 8), max_slots=8, compile_mode=None):
        self.T, self.pb, self.m, self.W = T, pb, m, T.W
        d, dt = T.d, T.dt; self.d, self.dt = d, dt
        self.buckets = sorted(seq_buckets); self.Smax = self.buckets[-1]; self.max_slots = max_slots
        self.cfg = pb.cfg; self.tc = pb.tc
        self.pad_id, self.eos_id = int(self.cfg.tts_pad_token_id), int(self.cfg.tts_eos_token_id)
        t0 = time.time()
        with torch.inference_mode():
            emb = pb.talker.get_text_embeddings().weight; V = emb.shape[0]
            tab = torch.empty(V, T.H, dtype=dt, device=d)
            for a in range(0, V, 16384):
                tab[a:a + 16384] = pb.talker.text_projection(emb[a:a + 16384])
            # the three special rows exactly as the eager path computes them (one 3-token call)
            tab[self.cfg.tts_bos_token_id] = pb.tts_bos[0, 0]; tab[self.eos_id] = pb.tts_eos[0, 0]; tab[self.pad_id] = pb.tts_pad[0, 0]
        self.tab = tab
        self.slots = {}; self.k = None; self.R = None                      # layout is fixed by the first registered voice
        self.graphs = {}; self.pool = None; self._cap = torch.cuda.Stream()
        cm = compile_mode if compile_mode is not None else os.environ.get("Q3_PF_COMPILE", "off")
        self.layer, self.post = self._layer, self._post
        if cm != "off":
            kw = {} if cm == "default" else {"mode": cm}
            self.layer = torch.compile(self._layer, dynamic=False, **kw); self.post = torch.compile(self._post, dynamic=False, **kw)
        self.boot_table_s = time.time() - t0
        self.st = dict(admits=0, seqs=0, pad_seqs=0, truncated=0)

    # ------------------------------------------------------------------ per (voice, language) prefix
    @torch.inference_mode()
    def voice_slot(self, voice, language="English", key=None):
        key = key or (id(voice), language.lower())
        if key in self.slots: return self.slots[key]
        T, pb, tc = self.T, self.pb, self.tc; lang = language.lower()
        if lang == "auto": pre = [[tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id]]
        else: pre = [[tc.codec_think_id, tc.codec_think_bos_id, tc.codec_language_id[lang], tc.codec_think_eos_id]]
        codec_in = torch.cat([pb._codec(pre), voice["spk"].view(1, 1, -1), pb._codec([[tc.codec_pad_id, tc.codec_bos_id]])], dim=1)
        role_ids = self._ids_cpu(pb.m._build_assistant_text("x"))[:3].to(self.d)
        role = self.tab[role_ids][None]
        head = torch.cat((pb.tts_pad.expand(-1, codec_in.shape[1] - 2, -1), pb.tts_bos), dim=1) + codec_in[:, :-1]
        ref_ids = voice["ref_ids"][0]; n_ref = int(ref_ids.shape[0]); codec = voice["codec"]; cl = int(codec.shape[1])
        assert n_ref < cl - 1, "reference text longer than the reference codes: no shared-prefix layout"
        prefix = torch.cat((role, head, self.tab[ref_ids][None] + codec[:, :n_ref]), dim=1)[0].to(self.dt)     # [k,H]
        k = int(prefix.shape[0]); L = 3 + head.shape[1] + cl; R = L - 1 - k
        if self.k is None:
            self.k, self.R, self.L = k, R, L; nl = len(self.W.layers)
            self.pK = torch.zeros(nl, self.max_slots, k, T.nkv, T.hd, dtype=self.dt, device=self.d)
            self.pV = torch.zeros_like(self.pK)
            self.crem = torch.zeros(self.max_slots, R + 1, T.H, dtype=self.dt, device=self.d)       # codec track of positions k..L-1
            self._static()
        assert (k, R) == (self.k, self.R), f"prefix layout {(k, R)} differs from {(self.k, self.R)}: use the eager path for this voice/language"
        slot = len(self.slots); assert slot < self.max_slots
        s = T.n_seats                                                         # compute the prefix K/V in the scratch seat
        T.prefill_hidden([s], [prefix])
        for l in range(len(self.W.layers)):
            self.pK[l, slot] = T.kc[l].view(-1, T.nkv, T.hd)[s * T.seat_len:s * T.seat_len + k]
            self.pV[l, slot] = T.vc[l].view(-1, T.nkv, T.hd)[s * T.seat_len:s * T.seat_len + k]
        self.crem[slot] = codec[0, n_ref:cl].to(self.dt)
        self.slots[key] = slot
        return slot

    def _static(self):
        T, d = self.T, self.d; S, R, k = self.Smax, self.R, self.k; mt = T.max_text
        self.s_seat = torch.full((S,), T.n_seats, dtype=torch.int64, device=d); self.s_slot = torch.zeros(S, dtype=torch.int64, device=d)
        self.s_rem = torch.full((S, R + 1), self.pad_id, dtype=torch.int64, device=d)
        self.s_trail = torch.full((S, mt + 1), self.pad_id, dtype=torch.int64, device=d); self.s_tlen = torch.zeros(S, dtype=torch.int64, device=d)
        self.h_seat = torch.zeros(S, dtype=torch.int64).pin_memory(); self.h_slot = torch.zeros(S, dtype=torch.int64).pin_memory()
        self.h_rem = torch.zeros(S, R + 1, dtype=torch.int64).pin_memory(); self.h_trail = torch.zeros(S, mt + 1, dtype=torch.int64).pin_memory()
        self.h_tlen = torch.zeros(S, dtype=torch.int64).pin_memory()
        self.z_seen = torch.zeros(S, self.W.vocab, dtype=torch.bool, device=d); self.z_i = torch.zeros(S, dtype=torch.int64, device=d)
        self.c_pos = torch.full((S,), k + R, dtype=torch.int64, device=d)
        self.ar = torch.arange(k + R, device=d)[None, :]
        self.cos = self.W.cos[k:k + R][None, :, None, :]; self.sin = self.W.sin[k:k + R][None, :, None, :]

    # ------------------------------------------------------------------ maths (same as Talker.prefill_hidden, dense [S,R])
    def _layer(self, x, pk, pv, ln1, qkv_w, qn, kn, cos, sin):
        T = self.T; S, R = x.shape[0], x.shape[1]; nq = T.nh * T.hd; nk = T.nkv * T.hd
        qkv = F.linear(rms(x, ln1), qkv_w)
        q = rope(rms(qkv[..., :nq].view(S, R, T.nh, T.hd), qn), cos, sin)
        kk = rope(rms(qkv[..., nq:nq + nk].view(S, R, T.nkv, T.hd), kn), cos, sin)
        v = qkv[..., nq + nk:].view(S, R, T.nkv, T.hd)
        return q, torch.cat((pk, kk), 1), torch.cat((pv, v), 1)

    def _post(self, x, o, o_w, ln2, gu, down):
        x = x + F.linear(o.reshape(x.shape[0], x.shape[1], -1), o_w)
        g = F.linear(rms(x, ln2), gu)
        return x + F.linear(F.silu(g[..., :self.T.ffn]) * g[..., self.T.ffn:], down)

    def _attn_ref(self, q, K, V):
        """float32 reference (flash-attn has no fp32 kernels): query i sees the k prefix keys and the new keys <= i."""
        T = self.T; S, R = q.shape[0], q.shape[1]; n = K.shape[1]
        m = torch.full((R, n), float("-inf"), device=q.device).triu(n - R + 1)
        qh = q.permute(0, 2, 1, 3).reshape(S, T.nkv, T.nh // T.nkv, R, T.hd); Kh = K.permute(0, 2, 1, 3)[:, :, None]; Vh = V.permute(0, 2, 1, 3)[:, :, None]
        a = torch.matmul(qh, Kh.transpose(-1, -2)) * T.scale + m
        return torch.matmul(torch.softmax(a, dim=-1), Vh).reshape(S, T.nh, R, T.hd).permute(0, 2, 1, 3)

    def _run(self, S):
        T, W = self.T, self.W; seat, slot = self.s_seat[:S], self.s_slot[:S]; R, k = self.R, self.k; nq = T.nh * T.hd
        T.seen.index_copy_(0, seat, self.z_seen[:S]); T.ngen.index_copy_(0, seat, self.z_i[:S])
        T.trail.index_copy_(0, seat, self.tab[self.s_trail[:S]]); T.trail_len.index_copy_(0, seat, self.s_tlen[:S])
        e = self.tab[self.s_rem[:S]] + self.crem[slot]                        # [S,R+1,H]: text track + codec track
        x = e[:, :R]
        flat = (seat[:, None] * T.seat_len + self.ar).reshape(-1)              # seat slots 0..k+R-1
        nl = len(W.layers)
        for l in range(nl):
            w = W.layers[l]
            q, K, V = self.layer(x, self.pK[l][slot], self.pV[l][slot], w.ln1, w.qkv, w.qn, w.kn, self.cos, self.sin)
            T.kc[l].view(-1, T.nkv, T.hd).index_copy_(0, flat, K.reshape(-1, T.nkv, T.hd))
            T.vc[l].view(-1, T.nkv, T.hd).index_copy_(0, flat, V.reshape(-1, T.nkv, T.hd))
            if l == nl - 1: break                                             # only the K/V of the last layer are needed
            if self.dt == torch.float32: o = self._attn_ref(q, K, V)
            else: o = flash_attn_func(q, K, V, softmax_scale=T.scale, causal=True)   # causal mask aligned bottom-right: query i sees prefix + new <= i
            x = self.post(x, o, w.o, w.ln2, w.gu, w.down)
        T.pos.index_copy_(0, seat, self.c_pos[:S])
        T.x_in.index_copy_(0, seat, e[:, R])
        T.pos[T.n_seats:].zero_()                                             # scratch seat (padded rows) stays at position 0

    @torch.inference_mode()
    def capture(self):
        assert self.k is not None, "register a voice first (voice_slot)"
        t0 = time.time()
        for S in reversed(self.buckets):
            self.s_seat.fill_(self.T.n_seats)
            for _ in range(3): self._run(S)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool, stream=self._cap):
                self._run(S)
            if self.pool is None: self.pool = g.pool()
            self.graphs[S] = g
        torch.cuda.synchronize()
        print(f"[fast-admit] prefix k={self.k} of L={self.L} (graphed R={self.R}), {len(self.graphs)} graphs in {time.time() - t0:.1f} s, text table {self.boot_table_s:.1f} s", flush=True)

    # ------------------------------------------------------------------ fast lane: prefill + frame 0 in ONE small graph
    @torch.inference_mode()
    def enable_first_frame(self, seed=12345, debug_logits=False):
        """Graphs (S = 1,2,4,8) that prefill the new seats AND produce their frame 0 (talker decode of the last prompt token,
        codebook 0 sampled, 15 code-predictor steps, next input written).  Everything the bulk frame graph also uses but
        cannot share while both run concurrently is private here: code-predictor K/V scratch, RNG, output buffer, memory
        pool, capture stream.  Per-seat state rows are disjoint from the bulk lane's as long as the engine adds the seats
        to the bulk list only after this graph has finished."""
        T, W, d = self.T, self.W, self.d; t0 = time.time()
        self.cp2 = CodePredictor(W, self.Smax, temperature=T.temperature, top_k=T.top_k)
        self.gen2 = torch.Generator(device=d); self.gen2.manual_seed(seed)
        self.o_codes = torch.zeros(self.Smax, 16, dtype=torch.int64, device=d)
        self.o_logits = torch.zeros(self.Smax, W.vocab, dtype=torch.float32, device=d) if debug_logits else None
        self.hp = torch.cuda.Stream(priority=-1)                             # high priority: pre-empts the bulk lane's kernels at kernel boundaries
        self.graphs_ff = {}; self.pool_ff = None; self._cap_ff = torch.cuda.Stream()
        for S in reversed(self.buckets):
            self.s_seat.fill_(T.n_seats)
            for _ in range(3): self._run_ff(S)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph(); g.register_generator_state(self.gen2)
            with torch.cuda.graph(g, pool=self.pool_ff, stream=self._cap_ff):
                self._run_ff(S)
            if self.pool_ff is None: self.pool_ff = g.pool()
            self.graphs_ff[S] = g
        T.ngen[T.n_seats] = 0; T.seen[T.n_seats] = False; torch.cuda.synchronize()
        print(f"[fast-admit] first-frame lane: {len(self.graphs_ff)} graphs in {time.time() - t0:.1f} s", flush=True)

    def _run_ff(self, S):
        T, W = self.T, self.W; self._run(S)
        seat = self.s_seat[:S]; pos = T.pos[seat]
        h = T.decode(T.x_in[seat], W.cos[pos][:, None, :], W.sin[pos][:, None, :], seat * T.seat_len + pos, (pos + 1).to(torch.int32), T.seat_pages[seat])
        T.pos.index_copy_(0, seat, pos + 1)
        lg = T._cb0_logits(h, seat)
        if self.o_logits is not None: self.o_logits[:S].copy_(lg)
        c0 = sample_topk(lg, T.temperature, T.top_k, self.gen2)
        g = T.ngen[seat]
        T.seen.index_put_((seat, c0), T.eos_col.new_ones(()).expand_as(c0)); T.ngen.index_copy_(0, seat, g + 1)
        codes, acc = self.cp2.run(h, c0, self.gen2)
        T.x_in.index_copy_(0, seat, acc + T.trail[seat, torch.minimum(g, T.trail_len[seat])])
        self.o_codes[:S].copy_(torch.cat([c0[:, None], codes], 1))
        T.pos[T.n_seats:].zero_()

    @torch.inference_mode()
    def admit_fast(self, seats, preps, stream=None):
        """Prefill + frame 0 for the new seats on the high-priority stream.  Asynchronous: returns a view of the static
        output [n,16] (codes of frame 0; column 0 == eos cannot happen, min 2 frames).  The caller records a CUDA event on
        the same stream to know when it is done, and must not call admit_fast again before that."""
        st = stream or self.hp
        with torch.cuda.stream(st):
            S = self._stage(seats, preps); self.graphs_ff[S].replay()
        return self.o_codes[:len(seats)]

    # ------------------------------------------------------------------ host side
    def _ids_cpu(self, s):
        ids = self.m.processor(text=s, return_tensors="pt", padding=True)["input_ids"]
        return ids[0] if ids.dim() == 2 else ids

    def prepare(self, text, language="English", slot=0):
        """CPU only (tokeniser).  Call it on the request thread; the result is consumed by admit()."""
        ids = self._ids_cpu(self.m._build_assistant_text(text))[3:-5].numpy()
        full = np.concatenate([ids, [self.eos_id]]); A = self.R + 1; mt = self.T.max_text
        p = Prep(); p.slot = slot; p.L = self.L
        p.rem = np.full(A, self.pad_id, np.int64); n = min(len(full), A); p.rem[:n] = full[:n]
        tr = full[A:]; p.truncated = len(tr) > mt; tr = tr[:mt]
        p.n_trail = len(tr); p.trail = tr
        return p

    @torch.inference_mode()
    def admit(self, seats, preps):
        """Replaces Q3Engine._prefill: KV of every prompt token but the last, pending input, trailing text, state reset."""
        S = self._stage(seats, preps); self.graphs[S].replay()

    def _stage(self, seats, preps):
        n = len(seats); S = next(b for b in self.buckets if b >= n); T = self.T
        self.h_seat[:S].fill_(T.n_seats); self.h_slot[:S].zero_(); self.h_tlen[:S].zero_()
        hr, ht = self.h_rem.numpy(), self.h_trail.numpy(); hr[:S] = self.pad_id; ht[:S] = self.pad_id
        for i, (s, p) in enumerate(zip(seats, preps)):
            self.h_seat[i] = s; self.h_slot[i] = p.slot; hr[i] = p.rem; ht[i, :p.n_trail] = p.trail; self.h_tlen[i] = p.n_trail
            self.st["truncated"] += p.truncated
        self.s_seat[:S].copy_(self.h_seat[:S], non_blocking=True); self.s_slot[:S].copy_(self.h_slot[:S], non_blocking=True)
        self.s_rem[:S].copy_(self.h_rem[:S], non_blocking=True); self.s_trail[:S].copy_(self.h_trail[:S], non_blocking=True)
        self.s_tlen[:S].copy_(self.h_tlen[:S], non_blocking=True)
        self.st["admits"] += 1; self.st["seqs"] += n; self.st["pad_seqs"] += S - n
        return S
