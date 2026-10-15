# flashinfer 0.7.1 V41 数值验证（里程碑 2）

- 日期：2026-10-15
- 状态：**V41 混合 cache kernel 数值验证通过**

---

## 验证设置

- 环境：隔离容器 `fiv072`（镜像 `dsv41-fp4kv:20261014`，含我们 5 patch + P0）
- flashinfer 0.7.1 + sm121a AOT（`sparse_mla_sm120.so` 2.5 MB 在位）
- shape：B=2, H=16, topk=64, PS=256, HD=512, NP=2, N=32
- 数据源：**同一个** `fused_store_cache` 写的 V41/V41_FP4 cache

## V41 混合 cache 的正确配方（实测得出）

```python
SparseMLASm120Wrapper(
    max_num_tokens=64, max_num_heads=16,
    kv_cache_format="fp8",          # ← 不是 'fp8_dsv41_fp4_ca'
    kv_scale_format="ue8m0_g32",    # ← 关键！DSV4_1 main cache 的标志
    extra_kv_fp4=True,              # ← 副缓存 FP4
    device=torch.device("cuda"))
```

错误配方会报：
```
ValueError: extra_kv_fp4 (V41_FP4 extra cache) requires
kv_cache_format='fp8' and kv_scale_format='ue8m0_g32' (a DSV4_1 main cache)
```

## run 签名（完整）

```python
w.run(q, kv_cache, indices, output, sm_scale, *,
      topk_length=None, attn_sink=None,
      extra_kv_cache=None, extra_indices=None, extra_topk_length=None,
      out_lse=None, mid_out=None, mid_lse=None,
      prefill_impl=None, return_lse=False, lse_scale=1.0)
```

## 数值结果

| 指标 | 值 | 判定 |
|---|---|---|
| LSE 差异 | 常量偏置 **2.1485 ± 0.0050** | ✅ 实现约定差异 |
| out cosine similarity | **0.999696** | ✅ 语义一致 |
| out max abs diff | 7.324e-03（scale 0.195） | ✅ FP8 量化误差量级 |
| out mean abs diff | 1.232e-03 | ✅ |
| mean \|out\| | triton 0.05073 vs flashinfer 0.05074 | ✅ 几乎相同 |

## LSE 偏置的性质

`lse_fi − lse_tri = 2.1485`（std 0.0050）⇒ **常量偏移**，不影响 softmax
归一化后的 `out`。这与子代理独立发现的"`topk_length=None` 时 LSE 有
+log2(topk) 偏置"同族（topk=64 时 log2(64)=6，此处 2.15 对应主副合并后的
有效 topk 约定差异）。

⇒ **不是数值错误，是实现约定差异**。接入时需确认 sglang 侧是否消费 LSE
（若消费，需统一约定；若不消费，无影响）。

---

## 结论

**flashinfer 0.7.1 的 V41 混合 cache kernel 在我们自己的代码路径上数值正确。**
加上上游测试 212/212 通过，升级路线的技术风险已基本排除。

剩余工作是"接进来"（工程集成），不是"能不能做"（技术可行性）。
