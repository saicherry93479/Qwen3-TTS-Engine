"""Talker on fixed seats: paged per-seat KV, one "frame" CUDA graph per batch bucket.

One frame for B live streams = talker decode of the pending input (28 layers, 1 token per stream) -> codebook 0 sampled on
the GPU (repetition penalty over the stream's own generated ids, suppressed specials, min 2 frames) -> the code predictor's
15 steps -> next talker input (16 summed code embeddings + next text token) written back to the seat. All per-seat state
lives on the GPU and is addressed by a seat-index tensor, so the only thing the host sends per step is the seat list.
Padded rows point at a scratch seat (index n_seats), never at a live one.
"""
import os, time
import numpy as np
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache, flash_attn_varlen_func
from .ops import rms, rope, mlp, sample_topk, lin
from .code_predictor import CodePredictor

NEG = float("-inf")


class Talker:
    def __init__(self, W, n_seats=256, seat_len=512, buckets=None, max_text=256, compile_mode=None, cp_compile=None,
                 temperature=0.9, top_k=50, rep_penalty=1.05, min_new=2, seed=0):
        assert seat_len % 256 == 0
        self.W = W; d, dt = W.device, W.dtype; self.d, self.dt = d, dt
        self.n_seats, self.seat_len, self.page, self.P = n_seats, seat_len, 256, seat_len // 256
        self.H, self.nh, self.nkv, self.hd, self.ffn = W.H, W.heads, W.kv_heads, W.head_dim, W.ffn
        self.scale = float(self.hd) ** -0.5
        self.temperature, self.top_k, self.rep, self.min_new = temperature, top_k, rep_penalty, min_new
        S1 = n_seats + 1                                                  # + scratch seat for padded rows
        self.kc = [torch.zeros(S1 * self.P, self.page, self.nkv, self.hd, dtype=dt, device=d) for _ in W.layers]
        self.vc = [torch.zeros(S1 * self.P, self.page, self.nkv, self.hd, dtype=dt, device=d) for _ in W.layers]
        kv_gb = 2 * len(W.layers) * self.kc[0].numel() * 2 / 2**30
        print(f"[talker] {W.size}: {n_seats} seats x {seat_len} positions, {len(W.layers)} layers, KV {kv_gb:.1f} GB", flush=True)
        # ---- per-seat state on the GPU
        self.pos = torch.zeros(S1, dtype=torch.int64, device=d)           # tokens in the seat's KV (= next position)
        self.x_in = torch.zeros(S1, self.H, dtype=dt, device=d)           # pending talker input
        self.seen = torch.zeros(S1, W.vocab, dtype=torch.bool, device=d)  # codebook-0 ids this stream generated
        self.ngen = torch.zeros(S1, dtype=torch.int64, device=d)          # frames generated
        self.max_text = max_text
        self.trail = torch.zeros(S1, max_text + 1, self.H, dtype=dt, device=d)   # trailing text embeddings, tts_pad at [len]
        self.trail_len = torch.zeros(S1, dtype=torch.int64, device=d)
        self.seat_pages = (torch.arange(S1, dtype=torch.int32, device=d)[:, None] * self.P + torch.arange(self.P, dtype=torch.int32, device=d)[None, :]).contiguous()
        sup = torch.zeros(W.vocab, device=d); sup[W.vocab - 1024:] = NEG; sup[W.eos] = 0.0
        self.suppress = sup
        self.eos_col = torch.zeros(W.vocab, dtype=torch.bool, device=d); self.eos_col[W.eos] = True
        self.one = torch.ones(1, dtype=torch.int64, device=d)
        self.gen = torch.Generator(device=d); self.gen.manual_seed(seed)
        # ---- buckets, static io
        self.buckets = sorted(set(buckets or [1, 2, 4, 8, 16, 32, 64, 128, 256]))
        self.max_bs = self.buckets[-1]
        self.s_seat = torch.full((self.max_bs,), n_seats, dtype=torch.int64, device=d)
        self.o_codes = torch.zeros(self.max_bs, 16, dtype=torch.int64, device=d)
        self.h_seat = torch.zeros(self.max_bs, dtype=torch.int64).pin_memory()
        self.cp = CodePredictor(W, self.max_bs, compile_mode=cp_compile, temperature=temperature, top_k=top_k)
        compile_mode = compile_mode if compile_mode is not None else os.environ.get("Q3_TALKER_COMPILE", "default")
        self.decode = self._decode
        if compile_mode != "off":
            kw = {} if compile_mode == "default" else {"mode": compile_mode}
            self.decode = torch.compile(self._decode, dynamic=False, **kw)
        self.graphs = {}; self.pool = None; self._cap = torch.cuda.Stream()   # private capture stream (cuBLAS workspace per owner)

    # ------------------------------------------------------------------ talker maths
    def _decode(self, x, cos, sin, slot, ctx, bt):
        """x [B,H] -> post-norm hidden [B,H]; appends each row's K/V at flat slot `slot` and attends over ctx tokens."""
        B = x.shape[0]; nq = self.nh * self.hd; nk = self.nkv * self.hd
        for l in range(len(self.W.layers)):
            w = self.W.layers[l]
            qkv = lin(rms(x, w.ln1), w.qkv)
            q = rope(rms(qkv[:, :nq].view(B, self.nh, self.hd), w.qn), cos, sin)
            k = rope(rms(qkv[:, nq:nq + nk].view(B, self.nkv, self.hd), w.kn), cos, sin)
            v = qkv[:, nq + nk:].view(B, self.nkv, self.hd)
            self.kc[l].view(-1, self.nkv, self.hd).index_copy_(0, slot, k)
            self.vc[l].view(-1, self.nkv, self.hd).index_copy_(0, slot, v)
            if self.dt == torch.float32:      # parity-only reference path (flash-attn has no fp32 kernels)
                o = self._attn_ref(q, l, slot, ctx)
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), self.kc[l], self.vc[l], cache_seqlens=ctx, block_table=bt, softmax_scale=self.scale, causal=True)
            x = x + lin(o.reshape(B, nq), w.o)
            x = x + mlp(rms(x, w.ln2), w, self.ffn)
        return rms(x, self.W.norm)

    def _attn_ref(self, q, l, slot, ctx):
        K = self.kc[l].view(-1, self.nkv, self.hd); V = self.vc[l].view(-1, self.nkv, self.hd); out = []
        for b in range(q.shape[0]):
            s0 = int(slot[b]) - (int(ctx[b]) - 1); k = K[s0:s0 + int(ctx[b])].transpose(0, 1); v = V[s0:s0 + int(ctx[b])].transpose(0, 1)   # [8,ctx,128]
            a = torch.matmul(q[b].view(self.nkv, self.nh // self.nkv, self.hd), k.transpose(-1, -2)) * self.scale
            out.append(torch.matmul(torch.softmax(a, dim=-1), v).reshape(self.nh, self.hd))
        return torch.stack(out, 0)

    def _cb0_logits(self, h, seat):
        """Processed codebook-0 scores (float32): repetition penalty, suppressed specials, EOS blocked for the first frames."""
        lg = lin(h, self.W.codec_head).float()
        lg = torch.where(self.seen[seat], torch.where(lg < 0, lg * self.rep, lg / self.rep), lg) + self.suppress
        return lg.masked_fill((self.ngen[seat] < self.min_new)[:, None] & self.eos_col[None, :], NEG)

    def _tail(self, h, seat):
        """hidden -> the frame's 16 codes; writes the next talker input and the sampling state back to the seats."""
        c0 = sample_topk(self._cb0_logits(h, seat), self.temperature, self.top_k, self.gen)
        g = self.ngen[seat]
        self.seen.index_put_((seat, c0), self.eos_col.new_ones(()).expand_as(c0))
        self.ngen.index_copy_(0, seat, g + 1)
        codes, acc = self.cp.run(h, c0, self.gen)
        t = self.trail[seat, torch.minimum(g, self.trail_len[seat])]
        self.x_in.index_copy_(0, seat, acc + t)
        return torch.cat([c0[:, None], codes], 1)

    def _frame(self, B):
        seat = self.s_seat[:B]; pos = self.pos[seat]
        h = self.decode(self.x_in[seat], self.W.cos[pos][:, None, :], self.W.sin[pos][:, None, :], seat * self.seat_len + pos,
                        (pos + 1).to(torch.int32), self.seat_pages[seat])
        self.pos.index_copy_(0, seat, torch.where(seat == self.n_seats, 0, pos + 1))   # padded rows (scratch seat) stay at position 0
        self.o_codes[:B].copy_(self._tail(h, seat))

    # ------------------------------------------------------------------ graphs
    @torch.inference_mode()
    def capture(self, buckets=None):
        t0 = time.time()
        for B in reversed(buckets or self.buckets):
            self.s_seat.fill_(self.n_seats); self.pos[self.n_seats] = 0
            for _ in range(3): self._frame(B); self.pos[self.n_seats] = 0          # warm: compile for this bucket, outside capture
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph(); g.register_generator_state(self.gen)
            with torch.cuda.graph(g, pool=self.pool, stream=self._cap):
                self._frame(B)
            if self.pool is None: self.pool = g.pool()
            self.graphs[B] = g
        self.pos[self.n_seats] = 0; self.ngen[self.n_seats] = 0; self.seen[self.n_seats] = False
        torch.cuda.synchronize()
        print(f"[talker] captured {len(self.graphs)} frame graphs in {time.time()-t0:.1f} s", flush=True)

    @torch.inference_mode()
    def step(self, seats):
        """One frame for the listed seats. Returns a view of the static output [B,16] (codes; column 0 == eos means done)."""
        B = len(seats); bs = next(b for b in self.buckets if b >= B)
        self.h_seat[:bs].fill_(self.n_seats); self.h_seat[:B] = torch.as_tensor(seats)
        self.s_seat[:bs].copy_(self.h_seat[:bs], non_blocking=True)
        self.graphs[bs].replay()
        return self.o_codes[:B]

    def set_seats(self, seats):
        """Bench helper: fix the seat list once, then call replay(bs) with no host work at all."""
        B = len(seats); bs = next(b for b in self.buckets if b >= B)
        self.s_seat[:bs].fill_(self.n_seats); self.s_seat[:B].copy_(torch.as_tensor(seats, device=self.d)); return bs

    # ------------------------------------------------------------------ prefill (eager, packed sequences)
    @torch.inference_mode()
    def prefill_hidden(self, seats, embeds):
        """embeds: list of [L_i, H] prompt embeddings. Writes the seats' KV, returns the last-token hidden [n, H]."""
        d = self.d; lens = [int(e.shape[0]) for e in embeds]; nq = self.nh * self.hd; nk = self.nkv * self.hd
        x = torch.cat(embeds, 0).to(self.dt); T = x.shape[0]
        pos = torch.cat([torch.arange(L, device=d) for L in lens])
        slot = torch.cat([s * self.seat_len + torch.arange(L, device=d) for s, L in zip(seats, lens)])
        cu = torch.tensor(np.concatenate([[0], np.cumsum(lens)]), dtype=torch.int32, device=d)
        cos, sin = self.W.cos[pos][:, None, :], self.W.sin[pos][:, None, :]
        for l in range(len(self.W.layers)):
            w = self.W.layers[l]
            qkv = F.linear(rms(x, w.ln1), w.qkv)
            q = rope(rms(qkv[:, :nq].view(T, self.nh, self.hd), w.qn), cos, sin)
            k = rope(rms(qkv[:, nq:nq + nk].view(T, self.nkv, self.hd), w.kn), cos, sin)
            v = qkv[:, nq + nk:].view(T, self.nkv, self.hd)
            self.kc[l].view(-1, self.nkv, self.hd).index_copy_(0, slot, k)
            self.vc[l].view(-1, self.nkv, self.hd).index_copy_(0, slot, v)
            if self.dt == torch.float32:      # parity-only reference path, one sequence
                assert len(lens) == 1
                qh = q.transpose(0, 1).view(self.nkv, self.nh // self.nkv, T, self.hd)
                a = torch.matmul(qh, k.transpose(0, 1)[:, None].transpose(-1, -2)) * self.scale + torch.full((T, T), NEG, device=d).triu(1)
                o = torch.matmul(torch.softmax(a, dim=-1), v.transpose(0, 1)[:, None]).reshape(self.nh, T, self.hd).transpose(0, 1)
            else:
                o = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max(lens), max_seqlen_k=max(lens), softmax_scale=self.scale, causal=True)
            x = x + F.linear(o.reshape(T, nq), w.o)
            x = x + mlp(rms(x, w.ln2), w, self.ffn)
        last = (cu[1:] - 1).long()
        return rms(x[last], self.W.norm)

    @torch.inference_mode()
    def admit(self, seats, embeds, trailing, pad_embed):
        """Prefill new streams and produce their first frame. trailing: list of [T_i,H] (text fed one token per frame);
        pad_embed [H] (tts_pad). Returns codes [n,16] of frame 0."""
        st = torch.as_tensor(seats, device=self.d)
        lens = torch.tensor([int(e.shape[0]) for e in embeds], device=self.d)
        assert int(lens.max()) + 64 <= self.seat_len
        self.seen[st] = False; self.ngen[st] = 0
        for s, tr in zip(seats, trailing):
            n = min(int(tr.shape[0]), self.max_text)
            self.trail[s, :n] = tr[:n].to(self.dt); self.trail[s, n] = pad_embed.to(self.dt); self.trail_len[s] = n
        h = self.prefill_hidden(seats, embeds)
        self.pos[st] = lens
        return self._tail(h, st)

    # ------------------------------------------------------------------ teacher-forced hooks (parity only)
    @torch.inference_mode()
    def tf_decode(self, seats, x):
        st = torch.as_tensor(seats, device=self.d); pos = self.pos[st]
        h = self._decode(x.to(self.dt), self.W.cos[pos][:, None, :], self.W.sin[pos][:, None, :], st * self.seat_len + pos, (pos + 1).to(torch.int32), self.seat_pages[st])
        self.pos[st] = pos + 1
        return h
