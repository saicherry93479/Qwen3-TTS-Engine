import os, json, functools
import torch
import torch.nn.functional as F

# OPT-3 switches (default off): Q3_GEMM_PLAN=1 splits a linear's rows by a measured plan (results/gemm_curve.json: GEMM time
# vs rows is a staircase, e.g. 288 rows of gate-up cost as much as 512, 256 + 32 is cheaper); each row is computed by one
# GEMM either way, so the result is the same maths (rounding-level).  Q3_SORT_TOPK=1 takes the k largest via a full sort
# (identical values and indices, 20 % faster at our shapes).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLAN_ON = os.environ.get("Q3_GEMM_PLAN", "0") == "1"
_SORT_TOPK = os.environ.get("Q3_SORT_TOPK", "0") == "1"
_CURVE = {}
if _PLAN_ON:
    try:
        _names = {"qkv06": (1024, 4096), "o06": (2048, 1024), "gu06": (1024, 6144), "down06": (3072, 1024), "head": (1024, 2048),
                  "codec_head06": (1024, 3072), "qkv17": (2048, 4096), "o17": (2048, 2048), "gu17": (2048, 12288), "down17": (6144, 2048),
                  "proj17": (2048, 1024), "codec_head17": (2048, 3072)}
        _raw = json.load(open(f"{_ROOT}/results/gemm_curve.json"))
        _CURVE = {_names[k]: {int(m): v for m, v in d.items()} for k, d in _raw.items() if k in _names}
    except Exception as e:
        print(f"[ops] no gemm plan: {e}")


@functools.lru_cache(None)
def _plan(K, N, M):
    c = _CURVE.get((K, N))
    if not c or M not in c: return (M,)
    @functools.lru_cache(None)
    def f(m):
        if m == 0: return 0.0, ()
        r = (c[m], (m,))
        for a in sorted(c):
            if a >= m: break
            v = c[a] + f(m - a)[0]
            if v < r[0] * 0.97: r = (v, (a,) + f(m - a)[1])        # only split for a real (> 3 %) gain
        return r
    return tuple(sorted(f(M)[1], reverse=True))


def lin(x, w, b=None):
    """F.linear with the measured row-split plan (Q3_GEMM_PLAN=1); identical to F.linear when the plan is one piece."""
    if _PLAN_ON and x.dim() == 2:
        ch = _plan(w.shape[1], w.shape[0], x.shape[0])
        if len(ch) > 1:
            out, a = [], 0
            for n in ch: out.append(F.linear(x[a:a + n], w, b)); a += n
            return torch.cat(out, 0)
    return F.linear(x, w, b)


def rms(x, w, eps: float = 1e-6):
    """Qwen3TTSRMSNorm: normalise in float32, cast back, then scale."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def rope(x, cos, sin):
    h = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., h:], x[..., :h]), dim=-1) * sin


def mlp(x, w, ffn: int):
    gu = lin(x, w.gu)
    return lin(F.silu(gu[..., :ffn]) * gu[..., ffn:], w.down)


def sample_topk(logits, temperature: float, k: int, gen):
    """HF order: temperature warp, keep the k largest, sample. logits float32 [B,V] -> ids [B]."""
    if _SORT_TOPK:
        v_, i_ = torch.sort(logits, dim=-1, descending=True); vals, idx = v_[:, :k], i_[:, :k]
    else:
        vals, idx = torch.topk(logits, k, dim=-1)
    p = torch.softmax(vals / temperature, dim=-1)
    j = torch.multinomial(p, 1, generator=gen)
    return idx.gather(1, j).squeeze(1)
