# flashinfer 升级路线 —— 执行进度

- 日期：2026-10-15
- 状态：**门控已扩展**，下一步是改 flashinfer 路径走 cake_dsv4

---

## 已完成

### 里程碑 1：0.7.1 在 GB10 上可用 ✅
- 4 个包装好（python 0.7.1 + cubin + jit_cache + sm121a AOT）
- 上游 `test_cake_dsv4.py` **212/212 通过**（0.14s，AOT 生效）
- AOT provider `sm121a` 在位，`sparse_mla_sm120.so` 2.5 MB

### 里程碑 2：V41 数值验证通过 ✅
| 指标 | 值 | 判定 |
|---|---|---|
| out cosine similarity | **0.999696** | ✅ |
| LSE 差异 | 常量偏置 2.1485 ± 0.0050 | ✅ 约定差异 |
| out max abs diff | 7.324e-03（scale 0.195） | ✅ FP8 误差量级 |

### 里程碑 3：patch 兼容性确认 ✅
**我们 5 个 patch 全部只碰 sglang 文件**，不碰 flashinfer：
| Patch | 目标 |
|---|---|
| M1 | `sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| M2 | `sglang/kernels/ops/attention/flash_mla_sm120.py` |
| M3+M4 | `sglang/srt/layers/attention/deepseek_v4_backend.py` |
| M5 | `sglang/kernels/ops/attention/flash_mla_sm120_triton.py` |
| M6 | `sglang/srt/model_executor/pool_configurator.py` |

⇒ **flashinfer 从单文件模块变包，不影响我们的 patch**（子代理的中风险评估过头）

**M6 已处理 kv_bytes**：`pool_configurator.py:1023-1026` 按布局走
（`528 fp8 main / 288 fp4 extra`）⇒ "容量 +0%" 风险点已解决

### 里程碑 4：门控扩展 ✅
`flash_mla_sm120_supports_v41_kv_layouts()` 现在：
```
backend=triton     → True
backend=flashinfer → True   ← 0.7.1 有 _MODEL_TYPE_DSV4_1 + cake_dsv4 + Wrapper 参数
其他               → False  ← fail closed
```
md5：`7e8a2137…` → `47be9703…`

---

## 关键 API（实测）

### V41 混合 cache 配方
```python
SparseMLASm120Wrapper(
    max_num_tokens=64, max_num_heads=16,
    kv_cache_format="fp8",          # 不是 'fp8_dsv41_fp4_ca'
    kv_scale_format="ue8m0_g32",    # ← DSV4_1 main 的标志
    extra_kv_fp4=True,
    device=torch.device("cuda"))
```

### run 签名
```python
w.run(q, kv_cache, indices, output, sm_scale, *,
      topk_length=None, attn_sink=None,
      extra_kv_cache=None, extra_indices=None, extra_topk_length=None,
      out_lse=None, mid_out=None, mid_lse=None,
      prefill_impl=None, return_lse=False, lse_scale=1.0)
```

### 0.7.1 的 V41 符号
```
_MODEL_TYPE_DSV4_1 = 5
_DECODE_DSV4_1_DISPATCH = _DecodeDispatchEnvelope(num_heads<=128, topk>=1)
_DECODE_DSV4_1_TOPK = 512
_KV_SCALE_FORMATS = {'auto', 'ue8m0_g32', 'pow2_fp32', 'arbitrary_fp32'}
```

---

## 剩余工作

| # | 任务 | 状态 |
|---|---|---|
| 1 | 放宽 `flash_mla_sm120.py` L356/L759 两处 584 断言 | 进行中 |
| 2 | 改 `_flash_mla_flashinfer` 走 `SparseMLASm120Wrapper`(cake) 而非 `_sparse_mla_sm120_paged_attention` | 待 |
| 3 | 四节点部署 + T0-T6 全门禁 | 待 |

---

## 环境

隔离容器 `fiv072`（镜像 `dsv41-fp4kv:20261014`）：
- flashinfer 0.7.1 全套 + sm121a AOT
- 我们 5 patch 在位
- 门控扩展已应用
- 备份：`/tmp/flash_mla_sm120.py.bak-pre-gate-ext`

宿主机 `/tmp/fiwork/`：源码 tarball + 4 个 wheel（共 ~1.75 GB）

---

## 生产状态

未受影响：V4 布局，`bpft=1670.75`，源码 md5 全基线，health 200。
