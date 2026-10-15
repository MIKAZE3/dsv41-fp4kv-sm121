# flashinfer 0.7.1 接入完成 —— V41 混合 cache 端到端打通

- 日期：2026-10-15
- 状态：**接入代码完成，decode/prefill 两条路径端到端验证通过**

---

## ★ 最终成果

通过 sglang 正式入口 `flash_mla_with_kvcache_sm120`，**flashinfer 后端**跑 V41 混合 cache：

```
backend = flashinfer, supports_v41 = True
out(2, 1, 16, 512) lse(2, 1, 16) finite=1.0000

★ 对比 Triton 参考
  out max abs diff = 7.324e-03 (scale 0.195)
  out cosine sim   = 0.999696
  lse diff: mean=2.1485 std=0.0194
  => MATCH
```

**prefill 路径也通**（`out(128,1,16,512) finite=1.0000`）。

---

## 做了什么（4 个改动，全部在 sglang 侧）

`flash_mla_sm120.py` md5 演进：
```
7e8a2137 (M2 patch 后)
  → 47be9703 (+门控扩展 DSV41_GATE_EXT)
  → 12bc329e (+cake route 断言放宽 + _flash_mla_flashinfer_dsv41)
  → 320e3158 (+decode route DSV41_DECODE_ROUTE)
  → f38e9b7c (+cache key 修复 DSV41_CACHE_KEY_FIX)
```

| 改动 | 内容 |
|---|---|
| **DSV41_GATE_EXT** | `flash_mla_sm120_supports_v41_kv_layouts()` 从"仅 triton"扩展为"triton 或 flashinfer≥0.7.1"；加 `_flashinfer_supports_dsv41()` 探针（fail-closed） |
| **DSV41_CAKE_ROUTE** | 放宽两处 584 断言：`_bpt in (584,528,288)`；非 584 且支持则分流到 `_flash_mla_flashinfer_dsv41` |
| **DSV41_DECODE_ROUTE** | `_flash_mla_flashinfer`（decode 路径）开头加 V41 分流 |
| **DSV41_CACHE_KEY_FIX** | wrapper cache key 含 `max_num_tokens`（2 的幂增长），否则大批次拿到过小的 wrapper |

新增函数 `_flash_mla_flashinfer_dsv41`：
```python
SparseMLASm120Wrapper(
    max_num_tokens=<2 的幂, >= num_tokens>,
    max_num_heads=num_heads,
    kv_cache_format="fp8",
    kv_scale_format="ue8m0_g32",     # ← DSV4_1 main 标志
    extra_kv_fp4=bool(extra_bpt),
    device=dev)
wrapper.run(q2, kv_u8, idx, out, sm_scale,
            topk_length=..., attn_sink=...,
            extra_kv_cache=..., extra_indices=..., extra_topk_length=...,
            out_lse=...)
```

---

## 关键 API（实测）

### V41 混合 cache 配方
```python
SparseMLASm120Wrapper(
    max_num_tokens=64, max_num_heads=16,
    kv_cache_format="fp8",          # 不是 'fp8_dsv41_fp4_ca'
    kv_scale_format="ue8m0_g32",
    extra_kv_fp4=True, device=torch.device("cuda"))
```

### 0.7.1 的 V41 符号
```
_MODEL_TYPE_DSV4_1 = 5
_DECODE_DSV4_1_DISPATCH = _DecodeDispatchEnvelope(num_heads<=128, topk>=1)
_DECODE_DSV4_1_TOPK = 512
_KV_SCALE_FORMATS = {'auto', 'ue8m0_g32', 'pow2_fp32', 'arbitrary_fp32'}
```

### 分发逻辑（sglang 入口）
```python
if _sm120_default_backend == "flashinfer":
    if q.shape[0] > SM120_DECODE_MAX_TOKENS:   # 64
        return _flash_mla_sm120_prefill(...)   # ← 已接 cake route
    return _flash_mla_flashinfer(...)          # ← 已接 cake route
```

---

## 剩余工作

| # | 任务 | 状态 |
|---|---|---|
| 1 | 把这 4 个改动固化为可复用的 patch（或并入 M2） | 进行中 |
| 2 | 四节点装 flashinfer 0.7.1 + AOT + 应用接入改动 | 待 |
| 3 | 停机窗口 T0-T6 全门禁 | 待 |

---

## 风险与对策

| 风险 | 状态 |
|---|---|
| flashinfer 升级破坏现有 V4 生产 | ✅ 默认路由不变；`backend=flashinfer` + `bpt=584` 走原路径 |
| 我们的 5 patch 受影响 | ✅ 全部只碰 sglang 文件，与 flashinfer 无关 |
| 容量增益 +0% | ✅ M6 patch 已让 `pool_configurator.py` 按布局走（528/288） |
| LSE 约定差异 | ⚠️ 常量 +2.15 偏置，不影响 softmax 后的 out；需确认 sglang 是否消费 LSE |
| 旧 flashinfer 误判 | ✅ 探针 fail-closed：缺符号即返回 False |

---

## 环境

- 隔离容器 `fiv072`（镜像 `dsv41-fp4kv:20261014`）
- flashinfer 0.7.1 + cubin + jit_cache + sm121a AOT
- 接入改动 4 处已应用，备份在 `/tmp/flash_mla_sm120.py.bak-pre-*`
- 宿主机 `/tmp/fiwork/`：源码 + 4 wheel（~1.75 GB）

---

## 生产状态

未受影响：V4 布局，`bpft=1670.75`，源码 md5 全基线，health 200。
