# flashinfer 0.7.1 V41 API 映射（实测）

- 日期：2026-10-15
- 来源：隔离容器 `fiv071` 实测 `dir()` + `inspect.signature`

---

## V41 混合 cache 的入口

### 1. 主 wrapper
```python
SparseMLASm120Wrapper(
    max_num_tokens=None, max_num_heads=None,
    *, d_v=512, kv_scale_format='auto',
    kv_cache_format='fp8',          # ← V41 混合：'fp8_dsv41_fp4_ca'
    extra_kv_fp4=False,             # ← V41 混合：True
    comp...                         # (被截断，需完整签名)
)
```

### 2. 主调用（cake 路径）
```python
run_cake_dsv4(
    *, query, swa_kv_cache, compressed_kv_cache, workspace_buffer,
       sparse_indices, sparse_topk_lens, ...
)
```
辅助：
```python
cake_dsv4_workspace_layout(num_query_tokens, num_heads, num_splits) -> WorkspaceLayout
cake_dsv4_workspace_reset(workspace_buffer) -> None
get_cake_dsv4_workspace_bytes(num_query_tokens, num_heads, sparse_topk, dtype, *, num_splits=None) -> int
resolve_cake_dsv4_sparse_metadata(sparse_indices, sparse_topk_lens=None, *,
                                  extra_sparse_indices=None, extra_sparse_topk_lens=None, ...)
```

### 3. DSV4 decode（非 cake）
```python
sparse_mla_sm120_decode_dsv4(
    q, kv_cache, indices, mid_out, mid_lse, output, out_lse, sm_scale,
    *, topk_len...
)
```

### 4. V41 FP4 量化（写 cache 用）
```python
dsv41_fp4_quantize_pack_sparse_mla_cache(latent_kv, *, kv_layout='HND') -> torch.Tensor
dsv41_fp4_quantize_append_sparse_mla_cache(latent_kv, slot_mapping, cache) -> None
```

### 5. 配置查询
```python
supported_sparse_mla_sm120_configs(*, kv_cache_format='fp8') -> dict[str, SparseMLASm120DecodeConfig]
calibrate_sparse_mla_sm120(device=None, *, heads=None, topks=None,
                           families=None, force=False, ...)
```

### 6. V41 常量与 dispatch 表
```python
_MODEL_TYPE_DSV4_1
_DECODE_DSV4_1_DISPATCH
_DECODE_DSV4_1_TOPK
```

---

## 与我们的代码的对应

| flashinfer 0.7.1 | 我们的代码 |
|---|---|
| `swa_kv_cache` | 主 cache（V41 布局，528 B/token） |
| `compressed_kv_cache` | extra cache（V41_FP4 布局，288 B/token） |
| `dsv41_fp4_quantize_append_*` | 我们的 `fused_store_cache(layout=V41_FP4)` |
| `kv_cache_format='fp8_dsv41_fp4_ca'` | `SGLANG_DSV4_KV_LAYOUT=v41` + `COMPRESSED=fp4` |
| `extra_kv_fp4=True` | 副缓存启用 FP4 |

---

## 待完成：完整数值验证

需要：
1. 拿到 `SparseMLASm120Wrapper` 完整签名（被截断）
2. 用 `run_cake_dsv4` 跑 V41 混合 cache
3. 与我们的 Triton 参考输出对比（bit-exact 或已知量化误差内）
