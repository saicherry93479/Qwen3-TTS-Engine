"""Load an upstream Qwen3-TTS Base checkpoint (bf16) and expose its weights in the layout our engine uses.

Nothing here changes the maths: QKV and gate/up projections are concatenated row-wise (one GEMM computes the same rows),
RoPE tables are produced by the upstream rotary modules themselves.
"""
import os, types, torch
import torch._dynamo as _dynamo

# One static torch.compile specialisation per batch bucket. Dynamo's default limit is 8 recompiles per function; past it
# the function SILENTLY RUNS EAGER (the VoxCPM2 lesson, 2026-10-03). Raise every limit well above what we capture.
for _name, _val in (("recompile_limit", 4096), ("cache_size_limit", 4096), ("accumulated_recompile_limit", 65536), ("accumulated_cache_size_limit", 65536)):
    if hasattr(_dynamo.config, _name):
        try: setattr(_dynamo.config, _name, max(getattr(_dynamo.config, _name), _val))
        except Exception: pass

ROOT = os.environ.get("QWEN3TTS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_upstream(size="0.6B", device="cuda:0", dtype=torch.bfloat16):
    """dtype=torch.float32 is for the maths-parity test only (the engine serves in bf16, as upstream does)."""
    from qwen_tts import Qwen3TTSModel
    m = Qwen3TTSModel.from_pretrained(os.path.join(ROOT, "weights", f"Qwen3-TTS-12Hz-{size}-Base"), device_map=device, dtype=dtype)
    m.model.eval()
    return m


def _layer_weights(layer):
    a = layer.self_attn
    return types.SimpleNamespace(
        ln1=layer.input_layernorm.weight.data, ln2=layer.post_attention_layernorm.weight.data,
        qkv=torch.cat([a.q_proj.weight.data, a.k_proj.weight.data, a.v_proj.weight.data], 0).contiguous(),
        qn=a.q_norm.weight.data, kn=a.k_norm.weight.data, o=a.o_proj.weight.data,
        gu=torch.cat([layer.mlp.gate_proj.weight.data, layer.mlp.up_proj.weight.data], 0).contiguous(), down=layer.mlp.down_proj.weight.data)


@torch.inference_mode()
def extract(m, max_pos=4096):
    """m: upstream Qwen3TTSModel wrapper. Returns a namespace of plain tensors (all on the model's device, bf16)."""
    core = m.model; talker = core.talker; cp = talker.code_predictor; tc = talker.config; cc = cp.config
    dev = next(talker.parameters()).device; dt = next(talker.parameters()).dtype
    W = types.SimpleNamespace(size=core.tts_model_size, H=tc.hidden_size, n_layers=tc.num_hidden_layers, heads=tc.num_attention_heads,
                              kv_heads=tc.num_key_value_heads, head_dim=tc.head_dim, ffn=tc.intermediate_size, vocab=tc.vocab_size,
                              eos=tc.codec_eos_token_id, n_groups=tc.num_code_groups, device=dev, dtype=dt, cfg=core.config)
    W.layers = [_layer_weights(l) for l in talker.model.layers]
    W.norm = talker.model.norm.weight.data
    W.codec_emb = talker.model.codec_embedding.weight.data          # [3072, H]
    W.codec_head = talker.codec_head.weight.data                    # [3072, H]
    x = torch.zeros(1, 1, tc.hidden_size, dtype=dt, device=dev)
    pos = torch.arange(max_pos, device=dev)[None, None, :].expand(3, 1, -1)
    cos, sin = talker.model.rotary_emb(x, pos)                      # [3,1,P,128]; the three axes are identical for audio/text
    assert torch.equal(cos[0], cos[1]) and torch.equal(cos[0], cos[2])
    W.cos, W.sin = cos[0, 0].contiguous(), sin[0, 0].contiguous()   # [P,128] bf16
    # code predictor
    W.cp_H = cc.hidden_size; W.cp_ffn = cc.intermediate_size; W.cp_vocab = cc.vocab_size
    W.cp_layers = [_layer_weights(l) for l in cp.model.layers]
    W.cp_norm = cp.model.norm.weight.data
    W.cp_emb = torch.stack([e.weight.data for e in cp.model.codec_embedding], 0).contiguous()      # [15, 2048, H]
    W.cp_head = torch.stack([h.weight.data for h in cp.lm_head], 0).contiguous()                   # [15, 2048, cp_H]
    p = cp.small_to_mtp_projection
    W.cp_proj_w = p.weight.data if isinstance(p, torch.nn.Linear) else None
    W.cp_proj_b = p.bias.data if isinstance(p, torch.nn.Linear) else None
    cos, sin = cp.model.rotary_emb(torch.zeros(1, 1, cc.hidden_size, dtype=dt, device=dev), torch.arange(16, device=dev)[None, :])
    W.cp_cos, W.cp_sin = cos[0].contiguous(), sin[0].contiguous()   # [16,128]
    return W
