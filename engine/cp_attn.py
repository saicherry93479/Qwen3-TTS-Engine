"""OPT-3: the code predictor's 16-slot attention as one Triton kernel (switch Q3_CP_TRITON=1).

The predictor attends one query token (16 heads, GQA 8 KV heads, head_dim 128) over a static 16-slot K/V scratch with
slots > p masked.  As two cuBLAS batched matmuls + softmax this is 160 small kernels per frame (wmma, 30-60 us each at
B = 272-576).  Here one program per (row, KV head) loads q [2,128], K/V [p+1,128] and writes o [2,128]:
scores and the softmax in float32, probabilities rounded to the model dtype before P.V as upstream does
(softmax(att.float()).to(dtype)), P.V accumulated in float32.  Same maths; rounding order differs (rounding-level).
"""
import torch, triton
import triton.language as tl


@triton.jit
def _cp_attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr, n_live, scale,
                    sq_b, sk_b, sk_h, sk_s, so_b,
                    G: tl.constexpr, S: tl.constexpr, D: tl.constexpr):
    b = tl.program_id(0); h = tl.program_id(1)
    g = tl.arange(0, G); s = tl.arange(0, S); dd = tl.arange(0, D)
    q = tl.load(q_ptr + b * sq_b + (h * G + g)[:, None] * D + dd[None, :]).to(tl.float32)                     # [G,D]
    kv_off = b * sk_b + h * sk_h + s[:, None] * sk_s + dd[None, :]
    live = s < n_live
    K = tl.load(k_ptr + kv_off, mask=live[:, None], other=0.0).to(tl.float32)                                 # [S,D]
    att = tl.sum(q[:, None, :] * K[None, :, :], axis=2) * scale                                                 # [G,S]
    att = tl.where(live[None, :], att, float("-inf"))
    m = tl.max(att, axis=1)
    p = tl.exp(att - m[:, None]); p = p / tl.sum(p, axis=1)[:, None]
    p = p.to(q_ptr.dtype.element_ty).to(tl.float32)                                                             # upstream rounds P to bf16
    V = tl.load(v_ptr + kv_off, mask=live[:, None], other=0.0).to(tl.float32)
    o = tl.sum(p[:, :, None] * V[None, :, :], axis=1)                                                          # [G,D]
    tl.store(o_ptr + b * so_b + (h * G + g)[:, None] * D + dd[None, :], o.to(o_ptr.dtype.element_ty))


def cp_attn(q, K, V, n_live, scale):
    """q [B,16,128] (or [B,2048]), K/V [B,8,16,128] (row stride may exceed the live batch), n_live = p + 1 -> o [B,2048]."""
    B = q.shape[0]; q = q.reshape(B, 16 * 128)
    o = torch.empty(B, 16 * 128, dtype=q.dtype, device=q.device)
    _cp_attn_kernel[(B, 8)](q, K, V, o, n_live, scale, q.stride(0), K.stride(0), K.stride(1), K.stride(2), o.stride(0),
                            G=2, S=16, D=128, num_warps=4)
    return o
