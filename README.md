# DeepSeek-V4.1-Flash FP4 KV Cache — SM121 (GB10) patches

Production-ready patches that switch DeepSeek-V4.1-Flash KV cache from the V4
layout (584 B/token) to the official V4.1 layouts — **528 B FP8 main (SWA) +
288 B FP4 compressed (c128)** — on 4× DGX Spark (GB10 / SM121, aarch64), served
by SGLang with **flashinfer 0.7.1 `cake_dsv4` kernel** (upstream PR #5983).

## Results (measured on the reference 4-node cluster)

| Metric | V4 baseline | V41 (Triton path) | **V41 (flashinfer cake)** |
|---|---|---|---|
| KV bytes per full token | 1670.75 | 912.25 | **912.25** |
| KV capacity (tokens) | 6.56 M | 12.6 M | **12.77 M (+94%)** |
| Decode step (median, 8 runs) | 32.98 ms | 69.62 ms (+111%) ❌ | **33.63 ms (+2.0%)** ✅ |
| qeval | 72/75 | — | **71/75** ✅ (both misses are measurement-boundary effects, see docs) |
| Garbled outputs (batchtest + 8-way soak) | 0 | 0 | **0** ✅ |
| Prefix-cache hit rate | 90% | 90% | **90%** ✅ |

Capacity **+94% for +2.0% step time**.

## How it works

```
SGLANG_DSV4_KV_LAYOUT=v41  SGLANG_DSV4_COMPRESSED_KV_LAYOUT=fp4
SGLANG_SM120_FLASHMLA_BACKEND=flashinfer
  └─ M7 gate: flash_mla_sm120_supports_v41_kv_layouts() → True (flashinfer ≥ 0.7.1)
      └─ _flash_mla_flashinfer_dsv41()
          └─ SparseMLASm120Wrapper(kv_cache_format="fp8",
                                   kv_scale_format="ue8m0_g32",
                                   extra_kv_fp4=True)
              └─ flashinfer 0.7.1 cake_dsv4 kernel (sm121a AOT, prebuilt)
                 upstream: flashinfer-ai/flashinfer PR #5983
```

Numeric verification vs an independent Triton reference: output cosine
similarity **0.999696**; LSE differs by a constant offset only (softmax
normalisation convention; LSE is not consumed by the caller).

## Repository layout

```
patches/   M1..M7 patches against sglang (apply via apply.sh, dependency-ordered)
docs/      deployment guide, feasibility study, quality report, numeric evidence
scripts/   verification & gate scripts (e2e numeric check, JIT gate, T6 timing)
```

Start with [docs/REPRODUCIBLE-DEPLOY-GUIDE.md](docs/REPRODUCIBLE-DEPLOY-GUIDE.md) —
it contains the full environment matrix, per-file checksums, build steps,
**four-gate image verification** (including a real JIT-compile gate), deployment
window runbook, a 10-entry pitfall list, and a one-chain rollback.

## Key gotchas (all encountered for real)

1. `git archive` tarballs of flashinfer **omit the 3rdparty submodules** —
   `flashinfer/data/{cutlass,cccl,spdlog}` install as empty packages and JIT
   compilation fails with `cutlass/arch/barrier.h: No such file or directory`.
2. The sm121a AOT wheel ships `fp4_quantization_121`, but the runtime resolves
   SM12x to **`fp4_quantization_120f`** (family variant, CUDA >= 12.9) → AOT miss
   → source compile. Fix: copy the `.so` under the `120f` name **and** add it to
   the provider `manifest.json` (path+name must both match).
3. `docker commit` freezes the build container's entrypoint — starting the build
   container with `--entrypoint sleep` produces an image whose entrypoint is
   `sleep`, so the server never boots. Rewrite with `--change` or a Dockerfile.
4. Pure import/md5 gates are **not enough** — gate on a real JIT compile + the
   upstream test suite + an end-to-end numeric check.

## Requirements

- 4× DGX Spark (GB10, sm121a), aarch64, CUDA 13.0, torch 2.13.0+cu130
- SGLang dev-dsv41 image (baseline ships flashinfer 0.6.18)
- flashinfer 0.7.1 (four wheels, see deploy guide §3)
- `MAX_TOTAL_TOKENS` raised (8 M hard-caps the new ~12.7 M capacity)

## Status

Deployed and verified on a 4-node production cluster (2026-10-15). Rollback is
a single config-file restore (documented).
