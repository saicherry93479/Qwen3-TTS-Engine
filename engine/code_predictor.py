"""The 15 autoregressive code-predictor steps per frame, for B streams, as one unit.

Upstream runs a nested HF generate() per frame: a 2-token prefill [talker hidden, codebook-0 embedding] then 14 single-token
steps through 5 layers, each step with its own embedding table and head (130 ms per frame, flat in batch). Here the same
maths runs as 16 single-token causal steps (token 0 alone, then token 1, ...: identical to the 2-token causal prefill) over
a static 16-slot K/V buffer with an additive mask, so every step has the same shapes: ONE compiled step function per batch
bucket, and the whole loop including sampling is captured inside the caller's CUDA graph.
"""
import os, torch
import torch.nn.functional as F
from .ops import rms, rope, mlp, sample_topk, lin


class CodePredictor:
    def __init__(self, W, max_bs, compile_mode=None, temperature=0.9, top_k=50):
        self.W = W; self.L = len(W.cp_layers); self.nh, self.nkv, self.hd = 16, 8, W.head_dim
        self.H = W.cp_H; self.ffn = W.cp_ffn; self.scale = float(W.head_dim) ** -0.5
        self.temperature, self.top_k = temperature, top_k
        d, dt = W.device, W.dtype
        self.K = torch.zeros(self.L, max_bs, self.nkv, 16, self.hd, dtype=dt, device=d)
        self.V = torch.zeros(self.L, max_bs, self.nkv, 16, self.hd, dtype=dt, device=d)
        m = torch.full((16, 16), float("-inf"), device=d).triu(1)                 # row p: slots > p are masked
        self.mask = m.to(dt).contiguous()
        self.pidx = [torch.tensor([p], device=d) for p in range(16)]
        compile_mode = compile_mode if compile_mode is not None else os.environ.get("Q3_CP_COMPILE", "default")
        self.pre, self.post = self._pre, self._post
        if compile_mode != "off":
            kw = {} if compile_mode == "default" else {"mode": compile_mode}
            self.pre = torch.compile(self._pre, dynamic=False, **kw); self.post = torch.compile(self._post, dynamic=False, **kw)

    def proj(self, x):
        return x if self.W.cp_proj_w is None else lin(x, self.W.cp_proj_w, self.W.cp_proj_b)

    # A layer is two compiled pieces with the K/V write between them done as a plain in-place index_copy_.  Inside one
    # compiled function the write is functionalised into a copy of the WHOLE [B,8,16,128] buffer (17 of 35 ms at B=400).
    # Layer weights are arguments, so one specialisation per batch bucket serves all five layers.
    def _pre(self, x, ln1, qkv_w, qn, kn, cos, sin):
        B = x.shape[0]; nq = self.nh * self.hd; nk = self.nkv * self.hd
        qkv = lin(rms(x, ln1), qkv_w)
        q = rope(rms(qkv[:, :nq].view(B, self.nh, self.hd), qn), cos, sin)
        k = rope(rms(qkv[:, nq:nq + nk].view(B, self.nkv, self.hd), kn), cos, sin)
        return q, k, qkv[:, nq + nk:].view(B, self.nkv, self.hd)

    def _post(self, x, q, K, V, mask, o_w, ln2, gu, down):
        B = x.shape[0]
        att = torch.matmul(q.view(B, self.nkv, self.nh // self.nkv, self.hd), K.transpose(-1, -2)) * self.scale + mask   # [B,8,2,16]
        o = torch.matmul(torch.softmax(att.float(), dim=-1).to(x.dtype), V).reshape(B, self.nh * self.hd)
        x = x + lin(o, o_w)
        gu_ = lin(rms(x, ln2), gu)
        return x + lin(F.silu(gu_[..., :self.ffn]) * gu_[..., self.ffn:], down)

    def _s(self, x, B, p):
        """x [B, cp_H] at slot p -> normed hidden [B, cp_H]; the static K/V buffers get slot p."""
        W = self.W; cos, sin, mask, pidx = W.cp_cos[p], W.cp_sin[p], self.mask[p], self.pidx[p]
        for l in range(self.L):
            w = W.cp_layers[l]; K = self.K[l, :B]; V = self.V[l, :B]
            q, k, v = self.pre(x, w.ln1, w.qkv, w.qn, w.kn, cos, sin)
            K.index_copy_(2, pidx, k.unsqueeze(2)); V.index_copy_(2, pidx, v.unsqueeze(2))
            x = self.post(x, q, K, V, mask, w.o, w.ln2, w.gu, w.down)
        return rms(x, W.cp_norm)

    def run(self, h, c0, gen):
        """h [B,H] talker hidden (post-norm), c0 [B] sampled codebook 0. Returns codes [B,15] and the 16 embeddings summed
        at talker width (the codec part of the next talker input)."""
        W = self.W; B = h.shape[0]
        self._s(self.proj(h), B, 0)
        e = W.codec_emb[c0]; embs = [e]; codes = []
        for i in range(15):
            hid = self._s(self.proj(e), B, i + 1)
            c = sample_topk(lin(hid, W.cp_head[i]).float(), self.temperature, self.top_k, gen)
            codes.append(c); e = W.cp_emb[i][c]; embs.append(e)
        return torch.stack(codes, 1), torch.stack(embs, 1).sum(1)

    def tf_logits(self, h, codes16):
        """Teacher-forced: the 15 logits [B,15,2048] (float32) given the frame's 16 codes. Parity only."""
        W = self.W; B = h.shape[0]; out = []
        self._s(self.proj(h), B, 0)
        e = W.codec_emb[codes16[:, 0]]
        for i in range(15):
            hid = self._s(self.proj(e), B, i + 1)
            out.append(F.linear(hid, W.cp_head[i]).float()); e = W.cp_emb[i][codes16[:, i + 1]]
        return torch.stack(out, 1)
