# DeepSeek-V4.1-Flash FP4 KV Cache — SM121 (GB10) patches

FP4 KV cache patches for the **MiaAI-Lab TP4 production line**
([MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks),
4× DGX Spark GB10 / SM121, aarch64): switch DeepSeek-V4.1-Flash KV cache from the
V4 layout (584 B/token) to the official V4.1 layouts — **528 B FP8 main (SWA) +
288 B FP4 compressed (c128)** — served by SGLang with the **flashinfer 0.7.1
`cake_dsv4` kernel** (upstream PR #5983).

> This work is built **on top of the MiaAI-Lab TP4 production line** and follows
> its conventions (`start-tp4.sh`, `.env.tp4`, `state-tp4/`, `logs-tp4/`).
> All baseline numbers below are from that line's published profile.

## Speed comparison vs the MiaAI-Lab TP4 baseline

Same 4× DGX Spark cluster, same sparkDash-style prompts (greedy, 256 completion
tokens), aggregate tok/s with per-stream in brackets:

| prompt type | MiaAI TP4 baseline<br>(V4 KV, 584 B/tok) | **This work**<br>(V41+FP4 KV, 912 B/tok avg) |
|---|---:|---:|
| prose c1 (1 stream) | 87.7 | 72.9* |
| prose c16 (16 streams, aggregate) | 342.7 | ~400 (observed) |
| prefill 16k–128k | 4.9k–5.0k tok/s | (not re-measured) |
| KV capacity | 6.56 M tokens | **12.77 M tokens** (**+94%**) |
| KV bytes per full token | 1670.75 | **912.25** (−45%) |
| qeval | 72/75 | 71/75 |

\* c1 figure from a 64-token smoke probe (72.9 tok/s), not the full sparkDash
protocol — treat as indicative. The headline win is **capacity**: the same
~12.2 GB KV budget that held 6.56 M V4 tokens now holds 12.77 M V4.1 tokens.

Decode step time: 32.98 → 33.63 ms (**+2.0%**) despite reading 45% fewer KV
bytes per token — the flashinfer cake_dsv4 kernel recovers most of that cost.

## Results detail (measured on the reference cluster)

| Metric | V4 baseline | V41 (Triton path) | **V41 (flashinfer cake)** |
|---|---|---|---|
| KV bytes per full token | 1670.75 | 912.25 | **912.25** |
| KV capacity (tokens) | 6.56 M | 12.6 M | **12.77 M (+94%)** |
| Decode step (median, 8 runs) | 32.98 ms | 69.62 ms (+111%) ❌ | **33.63 ms (+2.0%)** ✅ |
| qeval | 72/75 | — | **71/75** ✅ (both misses are measurement-boundary effects) |
| Garbled outputs (batchtest + soak) | 0 | 0 | **0** ✅ |
| Prefix-cache hit rate | 90% | 90% | **90%** ✅ |

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

## What each patch does

| patch | target file | purpose |
|---|---|---|
| M1 | `srt/mem_cache/deepseek_v4_memory_pool.py` | SM121 gate for V41 layouts (default off) |
| M2 | `kernels/.../flash_mla_sm120.py` | decode read-path layout parameterisation + fail-loud guards |
| M3+M4 | `srt/layers/attention/deepseek_v4_backend.py` | pass layout into the sm120 entry |
| M7 | `kernels/.../flash_mla_sm120.py` | **flashinfer 0.7.1 cake_dsv4 route** (gate + prefill/decode routes + wrapper cache key) |
| M6 | `srt/model_executor/pool_configurator.py` | price pools per layout (528/288) — without this the capacity gain is **+0%** |
| M5 | `kernels/.../flash_mla_sm120_triton.py` | Triton backend V41 support (fallback path) |

## Repository layout

```
patches/   M1..M7 patches against sglang (apply via apply.sh, dependency-ordered)
docs/      reproducible deploy guide, feasibility study, quality report, numeric evidence
scripts/   verification & gate scripts (e2e numeric check, JIT gate, T6 timing)
```

Start with [docs/REPRODUCIBLE-DEPLOY-GUIDE.md](docs/REPRODUCIBLE-DEPLOY-GUIDE.md) —
environment matrix, per-file checksums, build steps, **four-gate image
verification** (including a real JIT-compile gate), deployment window runbook,
a 10-entry pitfall list, and a one-chain rollback.

## Key gotchas (all encountered for real)

1. `git archive` tarballs of flashinfer **omit the 3rdparty submodules** —
   `flashinfer/data/{cutlass,cccl,spdlog}` install as empty packages and JIT
   compilation fails with `cutlass/arch/barrier.h: No such file or directory`.
2. The sm121a AOT wheel ships `fp4_quantization_121`, but the runtime resolves
   SM12x to **`fp4_quantization_120f`** (family variant, CUDA ≥ 12.9) → AOT miss
   → source compile. Fix: copy the `.so` under the `120f` name **and** add it to
   the provider `manifest.json` (path+name must both match).
3. `docker commit` freezes the build container's entrypoint — starting the build
   container with `--entrypoint sleep` produces an image whose entrypoint is
   `sleep`, so the server never boots. Rewrite with `--change` or a Dockerfile.
4. Pure import/md5 gates are **not enough** — gate on a real JIT compile + the
   upstream test suite + an end-to-end numeric check.
5. `MAX_TOTAL_TOKENS` hard-caps the pool: leaving it at the MiaAI default 8 M
   throws away most of the new capacity — raise it to 13.2 M.

## Requirements

- 4× DGX Spark (GB10, sm121a), aarch64, CUDA 13.0, torch 2.13.0+cu130
- MiaAI-Lab TP4 production image (`Dockerfile.canary-roce` line, flashinfer 0.6.18)
- flashinfer 0.7.1 (four wheels, see deploy guide §3)
- `MAX_TOTAL_TOKENS=13200000`

## Credits & status

Built on the [MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks)
TP4 production line and its published baseline numbers. Uses the
flashinfer-ai/flashinfer cake_dsv4 kernel (PR #5983). Deployed and verified on a
4-node production cluster (2026-10-15, running since). Rollback is a single
config-file restore (documented).
