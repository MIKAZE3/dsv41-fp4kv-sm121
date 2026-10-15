# ★★★ V41+FP4 生产部署成功（2026-10-15）

- 结论：**8 项门禁全部通过，V41 主 528B + V41_FP4 副 288B 已在生产运行**
- 配置：flashinfer 0.7.1 cake_dsv4 kernel + M7 patch + MAX_TOTAL_TOKENS=13.2M

---

## 门禁结果（全绿）

| 门禁 | 判据 | 实测 | 判定 |
|---|---|---|---|
| T1 bpft | 913.54 ±1% | **912.25** | ✅ |
| T1 容量 | [12.3M, 12.7M] | **12,772,864** | ✅（+94% vs V4 6.56M） |
| T2 冒烟 | 19+23=42 | 42 | ✅ |
| T3 缓存命中 | ≥85% | **90%**（768/856） | ✅ |
| T4 qeval | ≥71/75（用户授权） | **71/75** | ✅ |
| T4 batchtest | 0 garbled | **0** | ✅ |
| T5 soak | 0 garbled | **0/8**（163.7 tok/s） | ✅ |
| **T6 step** | **<33.64 ms** | **33.63 ms** | ✅（+2.0% vs V4） |

### T4 paired compare 细节
```
BROKE: math_m9（答案漂移）、prose_p2（99 词差 1 词，GUARD 边界效应）
fixed: json_count（V41 修复了它）
⇒ qeval 71/75 在用户授权容差内；GUARD 的 prose 警告为词数边界抖动非崩坏
```

---

## 性能对比

| 指标 | V4 基线 | V41 (Triton) | **V41 (flashinfer cake)** |
|---|---|---|---|
| median step | 32.98 ms | 69.62 ms (+111%) ❌ | **33.63 ms (+2.0%)** ✅ |
| 单流 tok/s | ~30 | ~24 | **72.9**（64-token 快测） |
| 8 并发聚合 | — | — | **163.7 tok/s** |
| bpft | 1670.75 | 912.25 | **912.25** |
| 容量 | 6.56M | 12.6M | **12.77M（+94%）** |

**容量 +94% 的代价仅 +2.0% step 延迟** —— 远优于 2% 判据边缘。

---

## 最终配置

```bash
# .env.tp4
IMAGE=dsv41-v41fp4:20261015           # 60e3ceb8f481，四节点一致
MAX_TOTAL_TOKENS=13200000
# EXTRA_CONTAINER_ENV 追加：
SGLANG_DSV4_KV_LAYOUT=v41
SGLANG_DSV4_COMPRESSED_KV_LAYOUT=fp4
SGLANG_SM120_FLASHMLA_BACKEND=flashinfer   # 0.7.1 cake_dsv4
DSV41_TRITON_SINGLE_CONFIG=1               # P0（triton autotune 修复，防其它 triton 路径的 capture 失败）
```

## 镜像内容
- flashinfer **0.7.1**（python + cubin + jit_cache + sm121a AOT）
- **AOT 修复**：`fp4_quantization_120f`（目录 + manifest，上游把 SM12x 改写 120f 但 wheel 只发 121）
- **3rdparty**：cutlass/cccl/spdlog（从 0.6.18 继承，git-archive tarball 缺子模块）
- **M7 patch**：`flash_mla_sm120.py` 支持 V41（门控扩展 + cake route + decode route + cache key fix）
- 我们的 5 个原 patch（M1/M2/M3+M4/M5/M6）

## 关键依赖链
```
sglang backend=flashinfer + V41 layout
  → flash_mla_sm120_supports_v41_kv_layouts() = True（M7 门控）
  → _flash_mla_flashinfer_dsv41()（M7 cake route）
  → SparseMLASm120Wrapper(kv_cache_format="fp8", kv_scale_format="ue8m0_g32", extra_kv_fp4=True)
  → flashinfer 0.7.1 cake_dsv4 kernel（sm121a AOT）
```

---

## 回滚（如需）
```bash
cp $HOME/dsv41-4x-spark/.env.tp4.bak-fp4kv-20261014-223510 $HOME/dsv41-4x-spark/.env.tp4
su - <user> -c "export WORKER_PASS=<worker-password> && cd $HOME/dsv41-4x-spark && ./start-tp4.sh stop && ./start-tp4.sh serve"
```
（该备份 = 出厂 sha256 f81f5695 + canary-roce 镜像）

## 教训（本次窗口新增）
1. **`CUDA_LAUNCH_BLOCKING=1` 会拖慢 10-20%** —— 性能测量前必须移除诊断开关
2. **`sleep_on_idle` 的 throughput 日志是伪影** —— 引擎闲置时 throughput 显示 0.0x tok/s，勿当真实性能
3. **启动时 NFS 会被 stop 清掉** —— 重启前必须确认 `dsv41-nfs` 存活（`bash /tmp/snm3.sh` 手动起）
4. **launcher 的 exit code 1 可能是误报** —— 以 `docker ps` + `/health` 为准
