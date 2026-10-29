"""Exact streaming decoder for the Qwen3-TTS 12 Hz codec (codes -> 24 kHz PCM), slot-based for many streams.

The upstream decoder is causal with zero lookahead, so a stream can be decoded incrementally if every stage keeps the
history it looks at:
  * pre_conv (k=3)                      : last 2 quantised frames
  * 8 transformer layers, window 72     : rotated keys + values of the last 71 frames per layer (a query sees itself + 71)
  * 2 x [transposed conv k=s=2 (stateless), ConvNeXt depthwise k=7] : last 6 samples each
  * decoder conv k=7                    : last 6 samples
  * 4 blocks [SnakeBeta, transposed conv k=2r stride r, 3 residual units (SnakeBeta, conv k=7 dilation 1/3/9, SnakeBeta, conv k=1)]
                                        : previous input sample of the transposed conv; last 6*dilation samples per unit
  * SnakeBeta, conv k=7                 : last 6 samples
A chunk of n frames therefore costs O(n); nothing is re-decoded.  The windowed re-decode used upstream
(`chunked_decode(chunk_size=300, left_context_size=25)`) is NOT exact: the eight stacked 72-frame windows reach 8*71 frames back.

API
    dec = StreamDecoder(tokenizer_dir, n_slots=512, dtype=torch.float32)
    dec.reset(seat_ids)                               # start of an utterance (LongTensor or list)
    pcm = dec.decode(seat_ids, codes)                 # codes [B,16,n] int64 -> pcm [B, n*1920] float32
    dec.capture(buckets=(64,128,...), chunks=(1,2,4)) # optional: torch.compile + one CUDA graph per (bucket, chunk)
    dec.prime(seat, ref_codes); snap = dec.snapshot(seat); dec.restore(seat_ids, snap)   # voice clone: reference as history

Switches (both default on, both pass the SNR test): DEC_NHWC=1 runs the conv stack as conv2d on channels-last [B,C,1,T]
(cuDNN's fp16 tensor-core kernels are NHWC; -18 %), DEC_RING=1 keeps K/V in a ring of RING frames written in place
(one pass over the K/V state per call instead of four).  A chunk may be at most RING - 71 = 17 frames.
Precision: fp16 (47-52 dB vs the fp32 one-shot decode; upstream's own bf16 decode is 27-31 dB).  Never bf16.
Cost at 256 streams, compiled + graph: 14.1 / 24.5 / 44.5 / 83.7 ms per call for chunks of 1 / 2 / 4 / 8 frames.
"""
import math, os, time
import torch, torch.nn as nn, torch.nn.functional as F

WIN = 72            # sliding window of the transformer (keys seen by a query, itself included)
PAST = WIN - 1
HOP = 1920
RING = 88           # K/V ring length per stream: 71 past frames + up to 16 new frames per call, written in place at pos % RING


def load_upstream_decoder(tokenizer_dir, device="cuda", dtype=torch.float32):
    from transformers import AutoConfig, AutoModel
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
    try:
        AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
        AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)
    except ValueError:
        pass
    m = AutoModel.from_pretrained(tokenizer_dir, dtype=dtype).to(device).eval()
    return m.decoder


def _snake_params(act, dtype):
    # SnakeBeta: x + 1/(exp(beta)+1e-9) * sin(x*exp(alpha))^2   -> precompute the two per-channel factors once
    a = torch.exp(act.alpha.detach().float())
    ib = 1.0 / (torch.exp(act.beta.detach().float()) + act.no_div_by_zero)
    return a.to(dtype)[None, :, None].contiguous(), ib.to(dtype)[None, :, None].contiguous()


class StreamCore(nn.Module):
    """Pure function of (codes, states) -> (pcm, new states) for one chunk of n frames; n is fixed per trace."""

    def __init__(self, dec, dtype):
        super().__init__()
        self.dtype = dtype
        self.nhwc = os.environ.get("DEC_NHWC", "1") == "1"
        self.ring = os.environ.get("DEC_RING", "1") == "1"
        # DEC_LAZY=1 (default off): the residual sums h + c1 + c2 + c3 of a block are never written out; each consumer
        # (the Snake in front of the next conv) reads the block input and the k=1 conv outputs directly, so Inductor inlines
        # add + Snake into the kernel that builds the next conv's padded input.  Same sums, different association.
        self.lazy = os.environ.get("DEC_LAZY", "0") == "1"
        cfg = dec.config
        q = dec.quantizer
        # RVQ tables with the projections folded: frame vector = sum_k table_k[code_k]  (computed in fp32 once)
        tabs = []
        for rvq in (q.rvq_first, q.rvq_rest):
            for layer in rvq.vq.layers:
                cb = layer._codebook
                emb = cb.embedding_sum.detach().float() / cb.cluster_usage.detach().float().clamp(min=cb.epsilon)[:, None]
                emb = layer.project_out(emb) if not isinstance(layer.project_out, nn.Identity) else emb
                emb = F.conv1d(emb.t()[None], rvq.output_proj.weight.detach().float())[0].t()    # [bins, 512]
                tabs.append(emb)
        self.register_buffer("rvq", torch.stack(tabs).to(dtype))                                # [16, bins, 512]
        self.register_buffer("rvq_off", (torch.arange(len(tabs)) * tabs[0].shape[0])[None, :, None])
        self.pre_conv = dec.pre_conv.conv
        tr = dec.pre_transformer
        self.tr = tr
        self.layers = tr.layers
        self.H = cfg.num_attention_heads; self.hd = cfg.head_dim; self.L = len(tr.layers)
        self.register_buffer("inv_freq", tr.rotary_emb.inv_freq.detach().float().clone())
        self.ups = dec.upsample
        d = dec.decoder
        self.conv0 = d[0].conv
        self.blocks = nn.ModuleList(list(d[1:1 + len(cfg.upsample_rates)]))
        self.rates = list(cfg.upsample_rates)
        self.fin_conv = d[-1].conv
        sn = []
        for b in self.blocks:
            sn.append(_snake_params(b.block[0], dtype))
            for ru in b.block[2:]:
                sn.append(_snake_params(ru.act1, dtype)); sn.append(_snake_params(ru.act2, dtype))
        sn.append(_snake_params(d[-2], dtype))
        for i, (a, ib) in enumerate(sn):
            self.register_buffer(f"sa{i}", a); self.register_buffer(f"sb{i}", ib)
        self.n_sn = len(sn)

    # ---- state layout -------------------------------------------------------------------------------------------
    def state_shapes(self):
        cfg = self.tr.config
        kv = (RING, self.L, self.H, self.hd) if self.ring else (self.L, self.H, PAST, self.hd)
        shp = [("pre", (cfg.codebook_dim, 2)), ("K", kv), ("V", kv)]
        for i in range(len(self.ups)):
            shp.append((f"cnx{i}", (cfg.latent_dim, 6)))
        shp.append(("conv0", (cfg.latent_dim, 6)))
        for bi, b in enumerate(self.blocks):
            cin = b.block[1].conv.in_channels; cout = b.block[1].conv.out_channels
            shp.append((f"b{bi}_tc", (cin, 1)))
            for ri, dil in enumerate((1, 3, 9)):
                shp.append((f"b{bi}_r{ri}", (cout, 6 * dil)))
        shp.append(("fin", (self.fin_conv.in_channels, 6)))
        return shp

    def _snake(self, x, i):
        return x + getattr(self, f"sb{i}") * torch.sin(x * getattr(self, f"sa{i}")).square()

    @staticmethod
    def _cconv(conv, x, st):
        inp = torch.cat((st, x), dim=-1)
        y = F.conv1d(inp, conv.weight, conv.bias, dilation=conv.dilation, groups=conv.groups)
        return y, inp[..., inp.shape[-1] - st.shape[-1]:]

    def _rms(self, norm, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
        return norm.weight * xf.to(x.dtype)

    def forward(self, codes, pos, *st):
        """codes [B,16,n] int64; pos [B] int64 = frames already decoded for the stream; st = states in state_shapes() order."""
        st = list(st); new = []
        B, _, n = codes.shape
        # RVQ decode (sum of 16 table rows) -> [B,512,n]
        x = F.embedding(codes + self.rvq_off, self.rvq.flatten(0, 1)).sum(1).transpose(1, 2).contiguous()
        x, s = self._cconv(self.pre_conv, x, st[0]); new.append(s)
        h = self.tr.input_proj(x.transpose(1, 2))                                   # [B,n,512]
        # rope for absolute positions pos..pos+n-1 (fp32), mask: window + rows that have fewer than 71 past frames
        p = (pos[:, None] + torch.arange(n, device=codes.device)[None]).float()     # [B,n]
        fr = p[:, :, None] * self.inv_freq[None, None, :]
        emb = torch.cat((fr, fr), dim=-1)
        cos = emb.cos()[:, None].to(h.dtype); sin = emb.sin()[:, None].to(h.dtype)  # [B,1,n,hd]
        if self.ring:
            # K/V live in a per-stream ring [B,RING,L,H,hd] (slot = frame % RING).  The gathered ring is only READ here; the
            # new columns are returned and written in place by the caller, so a call makes one pass over the K/V state, not four.
            Kg, Vg = st[1], st[2]
            ar = torch.arange(n, device=codes.device)
            pm1 = (pos - 1)[:, None]
            qj = pm1 - torch.remainder(pm1 - torch.arange(RING, device=codes.device)[None], RING)      # frame held by each slot
            a_ = (pos[:, None] + ar[None])[:, :, None]                                                # query frame [B,n,1]
            old_ok = (qj[:, None, :] >= 0) & (qj[:, None, :] >= a_ - PAST)                            # [B,n,RING]
            new_ok = (ar[None, :] <= ar[:, None])[None].expand(B, -1, -1)                             # [B,n,n]
            bias = torch.zeros((B, n, RING + n), dtype=torch.float32, device=codes.device).masked_fill(~torch.cat((old_ok, new_ok), -1), float("-inf"))[:, None]
            scale = self.hd ** -0.5
            nK = []; nV = []
            for li, layer in enumerate(self.layers):
                at = layer.self_attn
                r = self._rms(layer.input_layernorm, h)
                qv = at.q_proj(r).view(B, n, self.H, self.hd).transpose(1, 2)
                kv = at.k_proj(r).view(B, n, self.H, self.hd).transpose(1, 2)
                vv = at.v_proj(r).view(B, n, self.H, self.hd).transpose(1, 2)
                half = self.hd // 2
                rot = lambda t: torch.cat((-t[..., half:], t[..., :half]), dim=-1)
                qv = qv * cos + rot(qv) * sin
                kv = kv * cos + rot(kv) * sin
                Kl = Kg[:, :, li].permute(0, 2, 3, 1)                                                  # [B,H,hd,RING]
                Vl = Vg[:, :, li].transpose(1, 2)                                                      # [B,H,RING,hd]
                lg = torch.cat((torch.matmul(qv, Kl), torch.matmul(qv, kv.transpose(-1, -2))), dim=-1).float() * scale + bias
                P = torch.softmax(lg, dim=-1).to(h.dtype)
                o = torch.matmul(P[..., :RING], Vl) + torch.matmul(P[..., RING:], vv)
                o = at.o_proj(o.transpose(1, 2).reshape(B, n, self.H * self.hd))
                h = h + layer.self_attn_layer_scale.scale * o
                r = self._rms(layer.post_attention_layernorm, h)
                m = layer.mlp
                h = h + layer.mlp_layer_scale.scale * m.down_proj(F.silu(m.gate_proj(r)) * m.up_proj(r))
                nK.append(kv.transpose(1, 2)); nV.append(vv.transpose(1, 2))                           # [B,n,H,hd]
            new.append(torch.stack(nK, 2)); new.append(torch.stack(nV, 2))                             # [B,n,L,H,hd] new columns
        else:
            j = torch.arange(PAST + n, device=codes.device)[None, None, :]              # key index
            qi = torch.arange(n, device=codes.device)[None, :, None]                    # query index in the chunk
            valid = j >= (PAST - pos.clamp(max=PAST))[:, None, None]
            mask = ((j >= qi) & (j <= PAST + qi) & valid)[:, None]                      # [B,1,n,71+n]
            Kc, Vc = st[1], st[2]; nK = []; nV = []
            for li, layer in enumerate(self.layers):
                at = layer.self_attn
                r = self._rms(layer.input_layernorm, h)
                qv = at.q_proj(r).view(B, n, self.H, self.hd).transpose(1, 2)
                kv = at.k_proj(r).view(B, n, self.H, self.hd).transpose(1, 2)
                vv = at.v_proj(r).view(B, n, self.H, self.hd).transpose(1, 2)
                half = self.hd // 2
                rot = lambda t: torch.cat((-t[..., half:], t[..., :half]), dim=-1)
                qv = qv * cos + rot(qv) * sin
                kv = kv * cos + rot(kv) * sin
                Ka = torch.cat((Kc[:, li], kv), dim=2); Va = torch.cat((Vc[:, li], vv), dim=2)
                o = F.scaled_dot_product_attention(qv, Ka, Va, attn_mask=mask)
                o = at.o_proj(o.transpose(1, 2).reshape(B, n, self.H * self.hd))
                h = h + layer.self_attn_layer_scale.scale * o
                r = self._rms(layer.post_attention_layernorm, h)
                m = layer.mlp
                h = h + layer.mlp_layer_scale.scale * m.down_proj(F.silu(m.gate_proj(r)) * m.up_proj(r))
                nK.append(Ka[:, :, n:]); nV.append(Va[:, :, n:])
            new.append(torch.stack(nK, 1)); new.append(torch.stack(nV, 1))
        h = self.tr.output_proj(self._rms(self.tr.norm, h)).transpose(1, 2).contiguous()   # [B,1024,n]
        if self.nhwc:
            return self._convs_nhwc(h, st, new)
        si = 3
        for tc, cnx in self.ups:
            # transposed conv with kernel = stride = 2 has no overlap: it is one GEMM per frame (same maths, no cuDNN layout issue)
            w = tc.conv.weight                                                       # [in, out, 2]
            y2 = torch.matmul(h.transpose(1, 2), w.reshape(w.shape[0], -1))          # [B, n, out*2]
            y2 = y2.view(y2.shape[0], y2.shape[1], w.shape[1], 2) + tc.conv.bias[None, None, :, None]
            h = y2.permute(0, 2, 1, 3).reshape(y2.shape[0], w.shape[1], -1)          # [B, out, 2n]
            y, s = self._cconv(cnx.dwconv.conv, h, st[si]); new.append(s); si += 1
            y = cnx.pwconv2(F.gelu(cnx.pwconv1(cnx.norm(y.transpose(1, 2)))))
            h = h + (cnx.gamma * y).transpose(1, 2).contiguous()
        h, s = self._cconv(self.conv0, h, st[si]); new.append(s); si += 1
        k = 0
        for bi, b in enumerate(self.blocks):
            r_ = self.rates[bi]
            a = self._snake(h, k); k += 1
            tc = b.block[1].conv
            inp = torch.cat((st[si], a), dim=-1)                                     # previous input + chunk
            y = F.conv_transpose1d(inp, tc.weight, tc.bias, stride=r_)               # length (m+2)*r
            h = y[..., r_: r_ + a.shape[-1] * r_]
            new.append(a[..., -1:]); si += 1
            for ru in b.block[2:]:
                a1 = self._snake(h, k); k += 1
                y, s = self._cconv(ru.conv1.conv, a1, st[si]); new.append(s); si += 1
                y = self._snake(y, k); k += 1
                h = h + F.conv1d(y, ru.conv2.conv.weight, ru.conv2.conv.bias)
        a = self._snake(h, k)
        y, s = self._cconv(self.fin_conv, a, st[si]); new.append(s)
        return (y[:, 0].float().clamp(min=-1, max=1), *new)

    # Same conv stack on [B,C,1,T] tensors in channels_last memory: cuDNN's fp16 tensor-core kernels are NHWC, so the 3-D path
    # pays an NCHW<->NHWC copy around every conv (9 of 56 ms at 256 streams x 4 frames).  Identical arithmetic.
    def _cconv4(self, conv, x, st):
        inp = torch.cat((st[:, :, None, :], x), dim=-1)
        y = F.conv2d(inp, conv.weight[:, :, None, :], conv.bias, dilation=(1, conv.dilation[0]), groups=conv.groups)
        return y, inp[:, :, 0, inp.shape[-1] - st.shape[-1]:]

    def _snake4(self, x, i):
        return x + getattr(self, f"sb{i}")[..., None] * torch.sin(x * getattr(self, f"sa{i}")[..., None]).square()

    def _convs_nhwc(self, h, st, new):
        CL = torch.channels_last
        h = h[:, :, None, :].contiguous(memory_format=CL)                              # [B,1024,1,n]
        si = 3
        for tc, cnx in self.ups:
            w = tc.conv.weight                                                       # [in, out, 2]; kernel = stride: one GEMM
            y2 = torch.matmul(h[:, :, 0].transpose(1, 2), w.reshape(w.shape[0], -1))
            y2 = y2.view(y2.shape[0], y2.shape[1], w.shape[1], 2) + tc.conv.bias[None, None, :, None]
            h = y2.permute(0, 2, 1, 3).reshape(y2.shape[0], w.shape[1], 1, -1).contiguous(memory_format=CL)
            y, s = self._cconv4(cnx.dwconv.conv, h, st[si]); new.append(s); si += 1
            y = cnx.pwconv2(F.gelu(cnx.pwconv1(cnx.norm(y.permute(0, 2, 3, 1)))))    # [B,1,T,C]
            h = h + (cnx.gamma * y).permute(0, 3, 1, 2)
        h, s = self._cconv4(self.conv0, h, st[si]); new.append(s); si += 1
        k = 0
        for bi, b in enumerate(self.blocks):
            r_ = self.rates[bi]
            a = self._snake4(h, k); k += 1
            tc = b.block[1].conv
            inp = torch.cat((st[si][:, :, None, :], a), dim=-1)
            y = F.conv_transpose2d(inp, tc.weight[:, :, None, :], tc.bias, stride=(1, r_))
            h = y[..., r_: r_ + a.shape[-1] * r_]
            new.append((inp if self.lazy else a)[:, :, 0, -1:]); si += 1     # same values; default graph unchanged
            if self.lazy:
                h0 = h; cs = []
                for ru in b.block[2:]:
                    if cs:                                                       # h0 + c1 + ... written so that no partial sum is shared
                        e = cs[-1]
                        for c in reversed(cs[:-1]): e = e + c
                        e = e + h0
                    else:
                        e = h0
                    a1 = self._snake4(e, k); k += 1
                    y, s = self._cconv4(ru.conv1.conv, a1, st[si]); new.append(s); si += 1
                    y = self._snake4(y, k); k += 1
                    cs.append(F.conv2d(y, ru.conv2.conv.weight[:, :, None, :], ru.conv2.conv.bias))
                h = cs[-1]
                for c in reversed(cs[:-1]): h = h + c
                h = h + h0
                continue
            for ru in b.block[2:]:
                a1 = self._snake4(h, k); k += 1
                y, s = self._cconv4(ru.conv1.conv, a1, st[si]); new.append(s); si += 1
                y = self._snake4(y, k); k += 1
                h = h + F.conv2d(y, ru.conv2.conv.weight[:, :, None, :], ru.conv2.conv.bias)
        a = self._snake4(h, k)
        y, s = self._cconv4(self.fin_conv, a, st[si]); new.append(s)
        return (y[:, 0, 0].float().clamp(min=-1, max=1), *new)


class StreamDecoder:
    def __init__(self, tokenizer_dir=None, n_slots=512,
                 dtype=torch.float32, device="cuda", upstream=None, channels_last=False):
        self.device = device; self.dtype = dtype; self.n_slots = n_slots
        dec = upstream if upstream is not None else load_upstream_decoder(tokenizer_dir or os.path.join(os.environ.get("QWEN3TTS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "Qwen3-TTS-Tokenizer-12Hz"), device, dtype)
        self.upstream = dec
        self.core = StreamCore(dec, dtype).to(device).eval()
        self.shapes = self.core.state_shapes()
        S = n_slots + 1                                           # last row = scratch slot for bucket padding
        self.dummy = n_slots
        self.pos = torch.zeros(S, dtype=torch.long, device=device)
        self.state = [torch.zeros((S, *shp), dtype=dtype, device=device) for _, shp in self.shapes]
        self.fn = self.core
        self.graphs = {}

    def state_bytes_per_slot(self):
        return sum(math.prod(s) for _, s in self.shapes) * torch.empty((), dtype=self.dtype).element_size()

    @torch.inference_mode()
    def reset(self, seat_ids):
        idx = torch.as_tensor(seat_ids, device=self.device, dtype=torch.long)
        self.pos.index_fill_(0, idx, 0)
        for k, s in enumerate(self.state):
            if not (self.core.ring and k in (1, 2)):              # ring K/V need no clearing: pos = 0 masks every slot
                s.index_fill_(0, idx, 0)

    @torch.inference_mode()
    def _step(self, idx, codes):
        st = [s.index_select(0, idx) for s in self.state]
        out = self.fn(codes, self.pos.index_select(0, idx), *st)
        if self.core.ring:
            n = codes.shape[-1]
            slot = torch.remainder(self.pos.index_select(0, idx)[:, None] + torch.arange(n, device=idx.device)[None], RING)
            flat = (idx[:, None] * RING + slot).reshape(-1)
            for k in (1, 2):
                self.state[k].view(-1, *self.state[k].shape[2:]).index_copy_(0, flat, out[1 + k].reshape(-1, *self.state[k].shape[2:]))
        for k, (s, n_) in enumerate(zip(self.state, out[1:])):
            if not (self.core.ring and k in (1, 2)):
                s.index_copy_(0, idx, n_)
        self.pos.index_add_(0, idx, torch.full_like(idx, codes.shape[-1]))
        return out[0]

    @torch.inference_mode()
    def decode(self, seat_ids, codes):
        idx = torch.as_tensor(seat_ids, device=self.device, dtype=torch.long)
        B, _, n = codes.shape
        g = self._pick(B, n)
        if g is None:
            return self._step(idx, codes)
        g["idx"].fill_(self.dummy); g["idx"][:B] = idx
        g["codes"].zero_(); g["codes"][:B] = codes
        g["graph"].replay()
        return g["out"][:B]

    # ---- voice-clone priming -------------------------------------------------------------------------------------
    # Upstream (ICL cloning) decodes [reference codes + generated codes] and trims the reference audio, so the generated
    # frames are decoded WITH the reference as history.  prime() puts a seat in that state; snapshot()/restore() let the
    # engine compute it once per voice and copy it into a seat at admission (no GPU work beyond a row copy).
    @torch.inference_mode()
    def prime(self, seat, ref_codes, chunk=8):
        """ref_codes [16,T] or [T,16] int64.  Leaves `seat` as if the reference had just been decoded."""
        if ref_codes.shape[0] != 16:
            ref_codes = ref_codes.t()
        ref_codes = ref_codes.to(self.device).long()
        self.reset([seat]); idx = torch.tensor([seat], device=self.device)
        saved, self.graphs = self.graphs, {}                      # eager path: any length, runs once per voice
        fn, self.fn = self.fn, self.core
        for a in range(0, ref_codes.shape[1], chunk):
            self._step(idx, ref_codes[None, :, a:a + chunk])
        self.graphs = saved; self.fn = fn

    @torch.inference_mode()
    def snapshot(self, seat):
        return dict(pos=self.pos[seat].clone(), state=[s[seat].clone() for s in self.state])

    @torch.inference_mode()
    def restore(self, seat_ids, snap):
        idx = torch.as_tensor(seat_ids, device=self.device, dtype=torch.long)
        self.pos.index_fill_(0, idx, int(snap["pos"]))
        for s, v in zip(self.state, snap["state"]):
            s.index_copy_(0, idx, v[None].expand(idx.shape[0], *v.shape))

    # ---- compile + CUDA graphs ------------------------------------------------------------------------------------
    def _pick(self, B, n):
        best = None
        for (b, c), g in self.graphs.items():
            if c == n and b >= B and (best is None or b < best[0]):
                best = (b, g)
        return None if best is None else best[1]

    @torch.inference_mode()
    def capture(self, buckets=(1, 16, 64, 128, 256, 512), chunks=(1, 2, 4), compile=True, graphs=True, mode="default"):
        import torch._dynamo
        torch._dynamo.config.recompile_limit = 4096; torch._dynamo.config.cache_size_limit = 4096
        torch._dynamo.config.accumulated_recompile_limit = 65536
        if compile:
            self.fn = torch.compile(self.core, mode=mode, dynamic=False)
        if not graphs:
            return
        self.cap_stream = torch.cuda.Stream()                    # private capture stream (own cuBLAS/cuDNN workspaces)
        self.pool = torch.cuda.graph_pool_handle()               # own memory pool, never shared with the LM graphs
        saved = [s.clone() for s in self.state]; saved_pos = self.pos.clone()
        for n in chunks:
            for b in buckets:
                g = dict(idx=torch.full((b,), self.dummy, device=self.device, dtype=torch.long),
                         codes=torch.zeros((b, 16, n), device=self.device, dtype=torch.long))
                for _ in range(3):                               # warm-up: compile + cuDNN algorithm search
                    self._step(g["idx"], g["codes"])
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr, pool=self.pool, stream=self.cap_stream):
                    g["out"] = self._step(g["idx"], g["codes"])
                g["graph"] = gr
                self.graphs[(b, n)] = g
        for s, sv in zip(self.state, saved):
            s.copy_(sv)
        self.pos.copy_(saved_pos)
