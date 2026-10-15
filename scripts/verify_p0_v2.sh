#!/bin/bash
# P0 修复验证 v2：把 Autotuner 的 configs 缩成 1 个（跳过 benchmark）
set -uo pipefail
docker run -d --name p0f --gpus all --entrypoint sleep dsv41-fp4kv:20261014 infinity

cat > /tmp/p0f.py <<"PYEOF"
"""P0 修复 v2 验证：configs 缩成 1 个 => Autotuner 跳过 benchmark。

原理（Autotuner.run 源码）：
    if len(self.configs) > 1:
        ... benchmark() ...     # 含 do_bench => sync => 作废 capture
    else:
        config = self          # 直接跑，无 benchmark
"""
import torch
from sglang.kernels.ops.attention import flash_mla_sm120_triton as M
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache

AUTO = M._tiled_sparse_decode_kernel
ORIG_CONFIGS = list(AUTO.configs)
print(f"原始 configs ({len(ORIG_CONFIGS)}): "
      f"{[(c.kwargs, c.num_warps, c.num_stages) for c in ORIG_CONFIGS]}")

DEV = "cuda"; PS = 256; HD = 512; H = 16; B = 4; NP = 4; N = 64
layout = KVLayout.V41
bpt = layout.bytes_per_token; pb = layout.page_bytes(PS)
torch.manual_seed(1)
lat = torch.randn(N, HD, dtype=torch.bfloat16, device=DEV) * 0.4
c = torch.zeros(NP, pb, dtype=torch.uint8, device=DEV)
fused_store_cache(lat, c, torch.arange(N, dtype=torch.int64, device=DEV),
                  page_size=PS, type="flashmla", layout=layout)
v = c.view(NP, pb).view(NP, PS, pb // PS)[..., :bpt].unsqueeze(2)
q = torch.randn(B, 1, H, HD, dtype=torch.bfloat16, device=DEV) * 0.2

def make(TK):
    idx = torch.randint(0, N, (B, TK), dtype=torch.int32, device=DEV)
    tl = torch.full((B,), TK, dtype=torch.int32, device=DEV)
    return idx, tl

# ---- 基线：用完整 configs 跑一次，记录输出与自选 config ----
idx, tl = make(128)
o_base, l_base = M._run_triton_sparse_decode(q, v, idx, tl, HD ** -0.5)
torch.cuda.synchronize()
chosen = None
for k, val in AUTO.cache.items():
    chosen = val
    break
print(f"\n自选 config: {chosen.kwargs} warps={chosen.num_warps} stages={chosen.num_stages}")
print(f"baseline 输出: out{tuple(o_base.shape)} lse{tuple(l_base.shape)}")

# ---- 应用 P0：configs 缩成 1 个（用自选的那个）----
AUTO.configs = [chosen]
AUTO.cache.clear()
print(f"\n[P0 已应用] configs = {[(c.kwargs, c.num_warps, c.num_stages) for c in AUTO.configs]}  cache cleared")

# 1) bit-exact
o_fix, l_fix = M._run_triton_sparse_decode(q, v, idx, tl, HD ** -0.5)
torch.cuda.synchronize()
d = (o_base.float() - o_fix.float()).abs().max().item()
dl = (l_base.float() - l_fix.float()).abs().max().item()
print(f"\n=== 1) bit-exact ===\n  max abs diff: out={d:.3e} lse={dl:.3e}")
print(f"  => {'BIT-EXACT OK' if d == 0 and dl == 0 else 'DIFFERS'}")
print(f"  cache 大小（应仍为空，因为没走 benchmark）: {len(AUTO.cache)}")

# 2) capture on cold topk
print(f"\n=== 2) capture on cold topk ===")
for TK in (96, 160, 192, 256):
    idx, tl = make(TK)
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            M._run_triton_sparse_decode(q, v, idx, tl, HD ** -0.5)
        torch.cuda.synchronize()
        print(f"  topk={TK:4d}  capture OK")
    except Exception as e:
        print(f"  topk={TK:4d}  capture FAIL {type(e).__name__} {str(e)[:60]}")

# 3) replay consistency
print(f"\n=== 3) replay consistency ===")
idx, tl = make(128)
o_e, l_e = M._run_triton_sparse_decode(q, v, idx, tl, HD ** -0.5)
torch.cuda.synchronize()
g3 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g3):
    o_g, l_g = M._run_triton_sparse_decode(q, v, idx, tl, HD ** -0.5)
torch.cuda.synchronize()
g3.replay(); torch.cuda.synchronize()
print(f"  replay==eager: out={torch.equal(o_e, o_g)} lse={torch.equal(l_e, l_g)}")

print("\nP0F-VERIFY-DONE")
PYEOF
docker cp /tmp/p0f.py p0f:/tmp/
docker exec p0f python3 /tmp/p0f.py 2>&1 | tail -32
docker rm -f p0f >/dev/null
