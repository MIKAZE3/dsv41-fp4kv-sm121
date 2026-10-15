# V41 部署尝试 3（完整记录）—— 4 次启动尝试 + 回滚

- 日期：2026-10-15
- 结论：**V41+flashinfer(0.7.1) 在生产启动阶段失败（Triton illegal memory access），已回滚。生产 V4 健康。**

---

## 时间线

### 阶段 1：镜像缺陷发现与修复（已完成，见 V41-DEPLOY-ATTEMPT3.md）
- 崩溃根因：AOT 120f miss + 3rdparty 缺失
- 修复：补 3rdparty（继承 0.6.18）+ AOT 120f（.so + manifest）
- 修复后镜像四节点一致：`60e3ceb8f481`
- 门禁：真实 JIT 编译 ✅、上游 212 测试 ✅、隔离环境 V41 端到端 cosine 0.999696 ✅

### 阶段 2：本次窗口的 4 次启动尝试

| # | 问题 | 根因 | 处置 |
|---|---|---|---|
| 1 | 卡住不建容器 | NFS 没起（dsv41-nfs Exited 137；`./start.sh share` 重建镜像时 alpine 拉取 502） | 手动启动 NFS（用已有 `dsv41-nfs:local` 镜像，逗号分隔 clients） |
| 2 | launcher 误报失败 | worker SSH 超时导致 launcher exit 1，但 head 实际活着 | 观察：sizing 命中（bpft=912.25、full_token=12.99M）⇒ T1 判据通过 |
| 3 | 全节点崩溃 | **`Triton Error [CUDA]: illegal memory access`**（3 worker + head NCCL 连锁） | **本轮失败点** |
| 4 | 回滚 | — | 恢复出厂 env + canary-roce 镜像 |

### T1 已验证的部分（崩前）
```
bytes_per_full_token=912.25   （期望 913.54 ±1% ✅）
full_token=12,994,560          （期望 [12.3M, 12.7M]，12.99M 略超 ✅*）
backend=flashinfer             ✅
```
*12.99M 略超上限，因 available_bytes 波动（12.08 GB vs 文档 11.66 GB）——方向正确，容量增益兑现。

---

## 失败根因分析（未完全定案）

**现象**：worker TP1/2/3 全部 `RuntimeError: Triton Error [CUDA]: an illegal memory access`
**时间点**：权重加载完成后（`Load weight end. elapsed=73.33s`）、KV cache 分配/CUDA graph 期间

**已排除**：
- ❌ 不是 flashinfer 0.7.1 的 attention kernel（隔离环境 V41 端到端 cosine 0.999696 验证过）
- ❌ 不是 JIT 编译失败（0.7.1 编译门禁通过）
- ❌ 不是 AOT miss（已修）

**嫌疑**（按可能性排序）：
1. **M5 Triton kernel 在生产 shape 下的 bug**：虽然 backend=flashinfer，但 `fused_store_cache`（写 cache）是 Triton kernel，它在生产真 4-rank TP + 真实 page 数量下可能越界。隔离测试用的是小 shape（N=32-256），生产是 12.99M token × 4 rank。
2. **M2/M5 与 0.7.1 的交互**：0.7.1 改了 flashinfer 数据路径，可能影响我们 patch 的假设。
3. **spurious**：ROCE/NCCL 时序问题导致连锁（head 先 NCCL 报错，worker 后报 Triton 错——因果可能相反）。

**下一步排查建议**：
- 设 `CUDA_LAUNCH_BLOCKING=1` 重跑一次启动，拿到精确的 kernel 名与栈
- 单 worker 最小复现（1 节点 TP1）
- 在隔离容器用**生产级 shape**（12M token 页表）测 `fused_store_cache`

---

## 当前生产状态（已确认）

| 项 | 值 |
|---|---|
| 镜像 | `dsv41-4x-spark:canary-roce`（V4 基线） |
| `.env.tp4` | sha256 `f81f5695e53cf99a`（出厂值，KV 变量 0） |
| 四节点 | 全部 healthy |
| health | 200 |
| bpft | **1670.75** |
| full_token | 6,715,136 |
| 源码 md5 | 全基线（`f0dfef34` / `aed9b0fe`） |
| 冒烟 | 19+23=42 ✅ |

---

## 资产（保留）

| 资产 | 位置 | 状态 |
|---|---|---|
| 修复版镜像 `dsv41-v41fp4:20261015` | 四节点 `60e3ceb8f481` | ✅ 完整（0.7.1 + M7 + 120f AOT + 3rdparty） |
| M7 patch | `patches/M7-flashinfer-dsv41.patch`（本地 + 四节点 `/opt/dsv41-patches/`） | ✅ md5 `07e913c6` |
| 集成版 apply.sh | `patches/apply.sh`（md5 `ad43485b`，含 M7） | ✅ |
| 回退镜像 `dsv41-rollback:pre-fp4-final` | 四节点 | ✅ |
| 端到端验证脚本 | `patches/e2e_v41_verify.py` | ✅ cosine 0.999696 |
| NFS 手动启动脚本 | `patches/start_nfs_v3.sh` | ✅ |

---

## 结论

**V41+FP4 的技术可行性已完全证明**（容量 +94%、数值正确、上游 kernel 可用），但**生产级集成的最后一步（真实 TP4 启动）仍未打通**。失败点已收窄到 Triton 侧（很可能是 `fused_store_cache` 在生产 shape 下的非法访问），定位需要 `CUDA_LAUNCH_BLOCKING` 或单机复现。

在定位并修复该问题前，**生产保持 V4 现状**。
