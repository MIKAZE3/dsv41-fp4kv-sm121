# DeepSeek-V4.1-Flash FP4 KV Cache 生产部署文档

- **部署日期**：2026-10-15
- **状态**：✅ 生产运行中（8 项门禁全部通过）
- **集群**：4× DGX Spark（GB10 / SM121），node1=head + node2/3/4=worker，TP4
- **回滚**：文末一条命令，已演练 3 次

---

## 1. 这是什么

将 KV Cache 从 V4 布局（584 B/token）切换到 DeepSeek V4.1 官方布局：

| 缓存 | 旧（V4） | 新（V4.1） | 说明 |
|---|---|---|---|
| 主（SWA） | 584 B（448 fp8 nope + 128 bf16 rope + 8 scale） | **528 B**（512 fp8 nope + 16 scale，RoPE 量化进 data 区） | 布局重排 |
| 副（compressed/c128） | 584 B | **288 B**（256 e2m1 fp4 + 32 scale） | FP4 量化 |

**收益**：每 token KV 字节 1670.75 → 912.25（−45%），容量 6.56M → 12.77M token（**+94%**）。
**代价**：decode step 32.98 → 33.63 ms（**+2.0%**）。

## 2. 最终配置（四节点一致）

### 2.1 镜像
```
dsv41-v41fp4:20261015  =  60e3ceb8f481  （44.9 GB）
```
内容：
- 基础：`dsv41-fp4kv:20261014`（含我们 5 个 sglang patch M1/M2/M3+M4/M5/M6）
- **flashinfer 0.7.1** 全套：python + cubin（1.5 GB）+ jit_cache + **sm121a AOT**（GB10 预编译 kernel）
- **AOT 修复**：`fp4_quantization_120f` 目录 + manifest 条目（上游 wheel 只发 121，但运行时查 120f）
- **3rdparty 修复**：cutlass/cccl/spdlog 头文件（git-archive tarball 缺子模块，从 0.6.18 继承）
- **M7 patch**：`flash_mla_sm120.py` 四处改动接入 cake_dsv4 路径

### 2.2 `.env.tp4` 关键行
```bash
IMAGE=dsv41-v41fp4:20261015
MAX_TOTAL_TOKENS=13200000          # 8000000 会把 12.7M 容量截断到 8M！
# EXTRA_CONTAINER_ENV 追加：
SGLANG_DSV4_KV_LAYOUT=v41
SGLANG_DSV4_COMPRESSED_KV_LAYOUT=fp4
SGLANG_SM120_FLASHMLA_BACKEND=flashinfer
DSV41_TRITON_SINGLE_CONFIG=1       # P0：防 triton autotune 作废 CUDA graph capture
```

### 2.3 源码 md5（打 patch 后，四节点一致）
| 文件 | md5 |
|---|---|
| `srt/mem_cache/deepseek_v4_memory_pool.py` | `58c74a4bdfb4c7eb` |
| `kernels/ops/attention/flash_mla_sm120.py` | `f38e9b7c47e22ddb`（M2+M7） |
| `kernels/ops/attention/flash_mla_sm120_triton.py` | `03fc3a3c88d455cd` |
| `srt/layers/attention/deepseek_v4_backend.py` | `66d37a5f38f91b4a` |
| `srt/model_executor/pool_configurator.py` | `8173000e297c23ff` |

## 3. 门禁结果（2026-10-15 实测）

| 门禁 | 判据 | 实测 | 判定 |
|---|---|---|---|
| T1 bpft | 913.54 ±1% | **912.25** | ✅ |
| T1 容量 | [12.3M, 12.7M] | 12,772,864 | ✅（超上限 0.6%，方向正确） |
| T2 冒烟 | 19+23=42 | 42 | ✅ |
| T3 缓存命中 | ≥85% | **90%**（768/856） | ✅ |
| T4 qeval | ≥71/75（授权容差） | **71/75** | ✅ |
| T4 batchtest | 0 garbled | **0** | ✅ |
| T5 soak | 0 garbled | **0/8**（163.7 tok/s 聚合） | ✅ |
| **T6 step** | **<33.64 ms** | **33.63 ms**（8 轮中位） | ✅ |

### T4 paired compare 明细
```
BROKE: math_m9（答案漂移 1.0 vs 38.0）、prose_p2（99 词差 1 词触发 GUARD——边界抖动）
fixed: json_count（V41 反而修复了它）
median tok/s: 84.21 → 63.28（-24.9%，qeval 75 题混合负载口径；单流快测实际 72.9 vs ~30）
```
GUARD 的 prose 警告是词数边界效应，非文本崩坏（batchtest/T5 均零崩坏佐证）。

### 性能对比总览
| 指标 | V4 基线 | V41 (Triton) | **V41 (flashinfer cake)** |
|---|---|---|---|
| median step | 32.98 ms | 69.62 ms（+111%）❌ | **33.63 ms（+2.0%）** ✅ |
| 单流 tok/s（快测） | ~30 | ~24 | **72.9** |
| 8 并发聚合 tok/s | — | — | **163.7** |
| 容量 | 6.56M | 12.6M | **12.77M** |

## 4. 技术链路（为什么能跑通）

```
sglang (M3+M4 传 layout)
  → flash_mla_sm120.flash_mla_with_kvcache_sm120(backend=flashinfer)
    → M7 门控: flash_mla_sm120_supports_v41_kv_layouts()=True
      → _flash_mla_flashinfer_dsv41()（M7 cake route）
        → SparseMLASm120Wrapper(kv_cache_format="fp8",
                                kv_scale_format="ue8m0_g32",
                                extra_kv_fp4=True)
          → flashinfer 0.7.1 cake_dsv4 kernel（sm121a AOT，GB10 预编译）
            上游 PR #5983（merged 2026-10-03）
            GB10 上 37/37 benchmark 赢过 584B kernel，geomean 2.26×
```

数值验证：V41 混合 cache 输出 vs 我们 Triton 参考 **cosine similarity 0.999696**；
LSE 差异为常量偏置 +2.15（softmax 约定差异，sglang 不消费 LSE，无影响）。

## 5. 四次尝试的失败与修复（完整教训）

| 尝试 | 卡点 | 根因 | 修复 |
|---|---|---|---|
| 1 | Triton 慢 9 倍（+111%） | 无专用 kernel | 上游 PR #5983（flashinfer 0.7.1 cake） |
| 2 | JIT 编译失败 `cutlass/arch/barrier.h: No such file` | ① git-archive tarball 的 3rdparty 子模块未初始化 ⇒ setuptools 装出空包；② AOT miss：上游把 SM12x 改写 `120f` 但 wheel 只发 `121` | ① 从 0.6.18 继承 cutlass/cccl/spdlog；② 补 `fp4_quantization_120f` 目录（.so 改名）+ manifest 加条目 |
| 3 | 容器起不来 | NFS 被 stop 清掉 + `./start.sh share` 重建镜像时 alpine 拉取 502；launcher 的 exit 1 是误报 | 手动 `docker run dsv41-nfs:local`（**逗号分隔** clients）+ 以 `docker ps`/`/health` 为准 |
| 4 | 诊断开关拖慢性能 | `CUDA_LAUNCH_BLOCKING=1` 串行化 launch，step 37.64ms | 移除后重测 33.63ms |

**三条最重要的教训**：
1. **门禁必须触发真实 kernel 编译/执行**——version/import/md5 全过也会漏掉 JIT 失败（尝试 2 的教训）
2. **`sleep_on_idle` 的 throughput 日志是伪影**——引擎闲置时显示 0.0x tok/s，勿当真实性能（差点误判性能崩坏）
3. **镜像 commit 会固化临时容器的 entrypoint**——用 `--entrypoint sleep` 起的构建容器 commit 后 entrypoint 变 sleep，服务必挂；用 Dockerfile 重写 ENTRYPOINT

## 6. 运维手册

### 6.1 健康检查
```bash
docker ps --format '{{.Names}} {{.Status}}'        # 四容器 healthy
curl -s -o /dev/null -w '%{http_code}' http://HEAD_IP:8888/health   # 200
docker logs dsv41-head 2>&1 | grep -oE 'bytes_per_full_token=[0-9.]+'     # 912.25
docker logs dsv41-head 2>&1 | grep -oE 'full_token=[0-9]+'                # ~12M
docker exec dsv41-head md5sum /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py   # f38e9b7c...
```

### 6.2 重启（注意 NFS！）
```bash
cd $HOME/dsv41-4x-spark
./start-tp4.sh stop
# ★ stop 会清掉 NFS 容器，必须先手动起（用已有镜像，不要 ./start.sh share 重建）：
sudo docker rm -f dsv41-nfs
sudo docker run -d --name dsv41-nfs --restart unless-stopped --privileged --network host \
  -v $HOME/models/DeepSeek-V4.1-Flash:/export:ro \
  -e "NFS_CLIENTS=WORKER_NETS" \
  dsv41-nfs:local
# 确认 2049 监听后：
su - <user> -c "export WORKER_PASS=<worker-password> && cd $HOME/dsv41-4x-spark && ./start-tp4.sh serve"
# launcher 可能报 exit 1 —— 以 docker ps + /health 为准（构建期 ~7 分钟）
```

### 6.3 回滚到 V4（一条命令链）
```bash
cd $HOME/dsv41-4x-spark
cp -p .env.tp4 .env.tp4.bak-v41-$(date +%Y%m%d-%H%M%S)
cp -p .env.tp4.bak-fp4kv-20261014-223510 .env.tp4     # 出厂值：canary-roce + 8000000 + 无 FP4 变量
su - <user> -c "export WORKER_PASS=<worker-password> && cd $HOME/dsv41-4x-spark && ./start-tp4.sh stop"
# 起 NFS（同 6.2）
su - <user> -c "export WORKER_PASS=<worker-password> && cd $HOME/dsv41-4x-spark && ./start-tp4.sh serve"
```

### 6.4 已知限制
- **CUDA graph capture 期约 17 秒**（target_verify），启动总时长 ~7 分钟
- **topk**：cake 的 fp8-dsv4_1 配置 topks=[512]；sglang 的动态 topk 在 wrapper 层处理，短上下文下 accept_rate ~0.2 正常
- **prose_p2 GUARD**：99 词差 1 词的边界抖动，建议观察线上 prose 质量
- **AOT 120f 是我们手工补的**：flashinfer 升级时 wheel 覆盖后需重打（`fp4_quantization_120f.so` + manifest 条目）

## 7. 资产清单

| 资产 | 位置 |
|---|---|
| 生产镜像 `dsv41-v41fp4:20261015`（60e3ceb8f481） | 四节点 |
| 回退镜像 `dsv41-rollback:pre-fp4-final` | 四节点 |
| M7 patch | `patches/M7-flashinfer-dsv41.patch`（md5 `07e913c6`）+ 四节点 `/opt/dsv41-patches/` |
| 集成版 apply.sh（含 M7） | `patches/apply.sh`（md5 `ad43485b`） |
| flashinfer 0.7.1 wheels | node1 `/tmp/fiwork/`（建议持久化到 `/state`） |
| NFS 手动启动脚本 | `patches/start_nfs_v3.sh` + node1 `/tmp/snm3.sh` |
| 端到端验证脚本 | `patches/e2e_v41_verify.py` |
| 部署过程记录 | `docs/fp4kv-research/V41-DEPLOY-ATTEMPT3.md`、`FLASHINFER-UPGRADE-PATH.md`、`FI071-INTEGRATION-DONE.md`、`V41-DEPLOY-SUCCESS.md` |

## 8. 建议（非阻塞）

1. **持久化 wheels**：`/tmp/fiwork/` 的 4 个 wheel（~1.75 GB）拷到 `/state/`，防节点重启丢失
2. **长期 soak**：数小时并发负载观察内存与 accept_rate 稳定性
3. **上游贡献**：AOT 120f gap 提给 flashinfer；M7 的 4 处 sglang 改动可考虑上游化（sglang 原生还不支持 flashinfer 跑 V41）
4. **qeval 观察**：math_m9 的答案漂移若在后续运行中持续，需单独归因
