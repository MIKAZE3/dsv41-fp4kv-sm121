"""端到端 V41 验证 v2（用正确的 kwargs）。"""

import torch

torch.cuda.set_per_process_memory_fraction(0.2)

DEV = "cuda"
PS, HD, H, B, NP, N, TOPK = 256, 512, 16, 2, 2, 32, 64

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache

lay_m, lay_e = KVLayout.V41, KVLayout.V41_FP4
bpt_m, pb_m = lay_m.bytes_per_token, lay_m.page_bytes(PS)
bpt_e, pb_e = lay_e.bytes_per_token, lay_e.page_bytes(PS)

torch.manual_seed(1)
lm = torch.randn(N, HD, dtype=torch.bfloat16, device=DEV) * 0.4
cm = torch.zeros(NP, pb_m, dtype=torch.uint8, device=DEV)
fused_store_cache(lm, cm, torch.arange(N, dtype=torch.int64, device=DEV),
                  page_size=PS, type="flashmla", layout=lay_m)
le = torch.randn(N, HD, dtype=torch.bfloat16, device=DEV) * 0.4
ce = torch.zeros(NP, pb_e, dtype=torch.uint8, device=DEV)
fused_store_cache(le, ce, torch.arange(N, dtype=torch.int64, device=DEV),
                  page_size=PS, type="flashmla", layout=lay_e)

q = torch.randn(B, 1, H, HD, dtype=torch.bfloat16, device=DEV) * 0.2
im = torch.randint(0, N, (B, TOPK), dtype=torch.int32, device=DEV)
tm = torch.full((B,), TOPK, dtype=torch.int32, device=DEV)
ie = torch.randint(0, N, (B, TOPK), dtype=torch.int32, device=DEV)
te = torch.full((B,), TOPK, dtype=torch.int32, device=DEV)

vm = cm.view(NP, pb_m).view(NP, PS, pb_m // PS)[..., :bpt_m].unsqueeze(2)
ve = ce.view(NP, pb_e).view(NP, PS, pb_e // PS)[..., :bpt_e].unsqueeze(2)

KW = dict(
    q=q, k_cache=vm, indices=im, topk_length=tm, attn_sink=None,
    head_dim_v=HD, softmax_scale=HD ** -0.5,
    extra_k_cache=ve, extra_indices_in_kvcache=ie, extra_topk_length=te,
    kv_layout=lay_m, extra_kv_layout=lay_e,
)

import importlib
import os

import sglang.kernels.ops.attention.flash_mla_sm120 as fm

print("=" * 88)
print("① Triton 参考")
print("=" * 88)
os.environ["SGLANG_SM120_FLASHMLA_BACKEND"] = "triton"
importlib.reload(fm)
print(f"  backend = {fm._sm120_default_backend}")
o_t, l_t = fm.flash_mla_with_kvcache_sm120(**KW)
torch.cuda.synchronize()
print(f"  out{tuple(o_t.shape)} lse{tuple(l_t.shape)} "
      f"finite={torch.isfinite(o_t.float()).float().mean().item():.4f}")

print()
print("=" * 88)
print("② flashinfer 路径（decode，B=2 <= 64）")
print("=" * 88)
os.environ["SGLANG_SM120_FLASHMLA_BACKEND"] = "flashinfer"
importlib.reload(fm)
print(f"  backend = {fm._sm120_default_backend}, "
      f"supports_v41 = {fm.flash_mla_sm120_supports_v41_kv_layouts()}")
try:
    o_f, l_f = fm.flash_mla_with_kvcache_sm120(**KW)
    torch.cuda.synchronize()
    print(f"  out{tuple(o_f.shape)} lse{tuple(l_f.shape)} "
          f"finite={torch.isfinite(o_f.float()).float().mean().item():.4f}")

    a, b = o_t.float(), o_f.float()
    d = (a - b).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    dl = (l_t.float() - l_f.float()).abs()
    print()
    print("  ★ 对比")
    print(f"    out max abs diff = {d:.3e} (scale {a.abs().max():.3f})")
    print(f"    out cosine sim   = {cos:.6f}")
    print(f"    lse diff: mean={dl.mean():.4f} std={dl.std():.4f}")
    print(f"    => {'MATCH' if cos > 0.99 else 'DIFFERS'}")
except Exception as e:
    import traceback
    print(f"  FAIL: {type(e).__name__}: {str(e)[:250]}")
    traceback.print_exc()

print()
print("=" * 88)
print("③ flashinfer 路径（prefill，B=128 > 64 走 prefill 分支）")
print("=" * 88)
try:
    Bp = 128
    qp = torch.randn(Bp, 1, H, HD, dtype=torch.bfloat16, device=DEV) * 0.2
    imp = torch.randint(0, N, (Bp, TOPK), dtype=torch.int32, device=DEV)
    tmp = torch.full((Bp,), TOPK, dtype=torch.int32, device=DEV)
    KWp = dict(KW)
    KWp.update(q=qp, indices=imp, topk_length=tmp)
    o_p, l_p = fm.flash_mla_with_kvcache_sm120(**KWp)
    torch.cuda.synchronize()
    print(f"  prefill out{tuple(o_p.shape)} "
          f"finite={torch.isfinite(o_p.float()).float().mean().item():.4f}")
    print("  => prefill 路径也通" if torch.isfinite(o_p.float()).all() else "  => 有 NaN")
except Exception as e:
    print(f"  prefill FAIL: {type(e).__name__}: {str(e)[:200]}")

print()
print("E2E-V41-V2-DONE")
