# DeepSeek-V4.1-Flash FP4 KV Cache — 可复现部署指南

- 目标读者：需要在**另一套 4× DGX Spark (GB10/SM121) 集群**上复现本部署的工程师
- 前提：已运行 DeepSeek-V4.1-Flash 的 V4 基线（本文从该状态出发）
- 结果：KV Cache 切换到 V4.1 布局（主 528B + 副 288B FP4），容量 +94%，step +2.0%
- 参考：本文是"怎么做"；背景与原理见文末参考文档列表

---

## 0. 环境与版本（本验证环境，复现请对齐）

| 组件 | 版本 | 备注 |
|---|---|---|
| 硬件 | 4× NVIDIA GB10，aarch64 | compute capability (12,1) = sm121a |
| CUDA | 13.0 (V13.0.88) | 容器内 nvcc |
| torch | 2.13.0+cu130 | 容器内 |
| sglang | 0.0.0.dev0（dev-dsv41 分支构建） | 基线镜像内置 |
| flashinfer | **0.6.18 → 0.7.1**（本文核心升级） | |
| 基线镜像 | `dsv41-4x-spark:canary-roce`（86489b2c314d, 33.4GB） | 含 V4 运行环境 |
| 基础镜像 | `lmsysorg/sglang:dev-dsv41@sha256:3dbc3130...` | canary-roce 的 FROM |
| 模型 | `$HOME/models/DeepSeek-V4.1-Flash`（476G, 48 shards） | |
| 网络 | head=HEAD_IP，workers=.12/.13/.14；head→worker SSH 免密 | launcher 依赖 |
| 启动器 | `$HOME/dsv41-4x-spark/start-tp4.sh`（serve/stop/share/smoke） | |

> ⚠️ **the cluster 集群特有**：`start-tp4.sh` 是本集群的封装（多节点 SSH 编排、NFS 权重共享、
> ROCE 配置）。在其他集群复现时，需要等价的多节点启动方式；本文第 6 节的容器参数
> （image/env/mounts）是通用约定，编排层按你的环境适配。

---

## 1. 总览：要做的事（依依赖顺序）

```
阶段 A  准备 patch 集（6 个 patch + apply.sh）          —— 第 2 节
阶段 B  下载 flashinfer 0.7.1 四件套                     —— 第 3 节
阶段 C  构建新镜像（0.6.18→0.7.1 + 6 patch + 两项修复）  —— 第 4 节（关键）
阶段 D  验证镜像（含真实 JIT 编译门禁——必做！）          —— 第 5 节
阶段 E  部署窗口（停机→切镜像→设变量→启动→验收）          —— 第 6 节
阶段 F  门禁验收（T1–T6）                                —— 第 7 节
```

预计总耗时：半天（首次）；镜像构建可复用，重部署 ~1.5h。

---

## 2. 阶段 A：准备 patch 集

6 个 patch 全部针对 **sglang 源码**（`/sgl-workspace/sglang/python/sglang/` 下相对路径），
按依赖顺序应用（apply.sh 的 ORDER 已编码）：

| # | patch | 目标文件 | md5 | 大小 | 作用 |
|---|---|---|---|---|---|
| M1 | `M1-layout-gate.patch` | `srt/mem_cache/deepseek_v4_memory_pool.py` | `9eb653186c25fc71971e01640b41eb58` | 4.1K | 总开关：SM121 可选 V41 布局（默认关） |
| M2 | `M2-gather-dequant.patch` | `kernels/ops/attention/flash_mla_sm120.py` | `12dc01172b7dfcd5c5f85aeca7809328` | 17K | decode 读路径 layout 参数化 + flashinfer fail-loud 守卫 |
| **M7** | `M7-flashinfer-dsv41.patch` | `kernels/ops/attention/flash_mla_sm120.py` | `07e913c6d5138ad51390bf7d632982f5` | 10K | **接入 flashinfer 0.7.1 cake_dsv4**（门控扩展+cake route+decode route+cache key fix） |
| M3+M4 | `M3-m4-pass-layout.patch` | `srt/layers/attention/deepseek_v4_backend.py` | `d04cdbe49c4b1ca786ac19ee4f6f9577` | 1.7K | 把 layout 传进 sm120 入口 |
| M6 | `M6-pool-configurator.patch` | `srt/model_executor/pool_configurator.py` | `5c5f173bda77164317ba7add23e87942` | 11K | 按主/副布局计价（528/288） |
| M5 | `M5-triton-v41.patch` | `kernels/ops/attention/flash_mla_sm120_triton.py` | `bc81ea29b7be034f4db9f39c2782baf7` | 20K | Triton 后端支持 V41（备用路径） |

另有 `apply.sh`（md5 `ad43485b8cd7c138ccf4552e314c497b`）：依赖序应用 + 基线 md5 断言 + 备份 + 幂等 + DRYRUN。

**改动内容速览**（审 patch 时看什么）：
- M1：`select_dsv4_kv_layout()` 加 SM121 分支 + `resolve_compressed_kv_layout()` 白名单
- M2：`_gather_and_dequant` 加 `layout` 参数；flashinfer 路径对非 584B fail-loud
- M7（本方案核心，4 处）：
  1. `DSV41_GATE_EXT`：`flash_mla_sm120_supports_v41_kv_layouts()` 从"仅 triton"扩展为
     "triton 或 flashinfer≥0.7.1（探测 `_MODEL_TYPE_DSV4_1` + `run_cake_dsv4` + Wrapper 三参数，fail-closed）"
  2. `DSV41_CAKE_ROUTE`：prefill 路径的 584 断言放宽为 `(584, 528, 288)`，非 584 分流到新函数
  3. `DSV41_DECODE_ROUTE`：`_flash_mla_flashinfer`（decode）开头同样分流
  4. 新函数 `_flash_mla_flashinfer_dsv41`：构造 `SparseMLASm120Wrapper(kv_cache_format="fp8",
     kv_scale_format="ue8m0_g32", extra_kv_fp4=True)` 并调用 `wrapper.run(...)`
  5. `DSV41_CACHE_KEY_FIX`：wrapper 缓存 key 含 `max_num_tokens`（2 的幂增长）
- M6：`_extra_kv_bytes()` 对 (V41, V41_FP4) 对返回 288（否则容量 +0%）
- M5：Triton kernel 加 `IS_V41/IS_FP4/DATA_STRIDE/...` constexpr（备用/对照路径）


**放置位置**：四节点容器内 `/opt/dsv41-patches/`（本文用 docker cp 从 head 分发）。

---

**应用后校验 md5**（patch 叠加后的最终文件指纹，复现时比对）：

| 文件 | 应用后 md5 |
|---|---|
| `srt/mem_cache/deepseek_v4_memory_pool.py` | `58c74a4bdfb4c7eb...` |
| `kernels/ops/attention/flash_mla_sm120.py`（M2+M7） | `f38e9b7c47e22ddb...` |
| `kernels/ops/attention/flash_mla_sm120_triton.py` | `50cf18089ff9cd65...` |
| `srt/layers/attention/deepseek_v4_backend.py` | `66d37a5f38f91b4a...` |
| `srt/model_executor/pool_configurator.py` | `8173000e297c23ff...` |


---

## 3. 阶段 B：下载 flashinfer 0.7.1 四件套

来源：GitHub releases（PyPI 在本网络不可达；`git archive` tarball 缺 3rdparty 子模块，见 §5 注意事项 1）。

```bash
# 在能访问 GitHub 的机器上（本文在 head 节点宿主机执行）
mkdir -p /tmp/fiwork && cd /tmp/fiwork

# 3.1 源码 tarball（57 MB）
curl -sSL -o fi.tar.gz \
  "https://github.com/flashinfer-ai/flashinfer/archive/refs/tags/v0.7.1.tar.gz"
# sha256 0cee9f4b51dc06729c7f103b0244e35927c83a287c6431898ab0b74759b8fdb7

# 3.2 cubin 包（1.5 GB）
curl -sSL -o flashinfer_cubin-0.7.1-py3-none-any.whl \
  "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.7.1/flashinfer_cubin-0.7.1-py3-none-any.whl"
# sha256 502eaa5f71e768950ada429618481a886388fcb34a5da6a2258979713cbaa327...

# 3.3 jit_cache 元数据包（4 KB，必须与主包版本一致，否则版本校验报错）
curl -sSL -o flashinfer_jit_cache-0.7.1+cu130-cp39-abi3-manylinux_2_28_aarch64.whl \
  "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.7.1/flashinfer_jit_cache-0.7.1+cu130-cp39-abi3-manylinux_2_28_aarch64.whl"
# sha256 759f35213fca2aca68d9afecf324888fdf95f20628e8a944dde031bcbbed25de...

# 3.4 sm121a AOT 包（175 MB，GB10 预编译 kernel —— 本方案的性能来源）
curl -sSL -o flashinfer_jit_cache_sm121a-0.7.1+cu130-cp39-abi3-manylinux_2_28_aarch64.whl \
  "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.7.1/flashinfer_jit_cache_sm121a-0.7.1+cu130-cp39-abi3-manylinux_2_28_aarch64.whl"
# sha256 712231a38b301cfdb507045f125ce07906bbb6b97ec05f87a3a2d353fea0aaa5...
```

**为什么要这四件**：0.7.1 起 flashinfer 拆成 python/cubin/jit-cache(通用元数据)/jit-cache-{arch}(预编译) 四包，
缺一不可（缺 cubin → import 失败；缺通用 jit-cache → `__version__` 属性缺失报错；缺 sm121a → 无预编译 kernel）。
**依据**：上游 PR [#5983](https://github.com/flashinfer-ai/flashinfer/pull/5983)（merged 2026-10-03），
标题即"SM120/SM121 DeepSeek-V4.1 mixed-cache sparse-MLA decode (528 B FP8 SWA + 288 B FP4 compressed)"，
GB10 上 37/37 benchmark 赢过 584B kernel（geomean 2.26×）。

---

## 4. 阶段 C：构建新镜像（核心，最容易踩坑）

**思路**：从基线镜像起临时容器 → 装 0.7.1 → apply 6 patch → **两项修复**（AOT 120f + 验证
3rdparty）→ commit。以下脚本在 head 节点宿主机以 root 执行：

```bash
#!/bin/bash
# build_v41fp4.sh —— 在 head 节点执行
set -euo pipefail
BASE=dsv41-4x-spark:canary-roce      # 或含 M1-M6 的 dsv41-fp4kv:20261014
NEW=dsv41-v41fp4:20261015
CONT=v41build
FW=/tmp/fiwork                        # 第 3 节下载的四件套
PATCH_DIR=/opt/dsv41-patches          # 宿主机上放 6 patch + apply.sh 的目录

docker rm -f $CONT >/dev/null 2>&1
docker run -d --name $CONT --gpus all --entrypoint sleep $BASE infinity

# --- 4.1 拷入资产 ---
docker cp $FW/fi.tar.gz $CONT:/tmp/
docker cp $FW/flashinfer_cubin-0.7.1-py3-none-any.whl $CONT:/tmp/
docker cp $FW/flashinfer_jit_cache-0.7.1+cu130-cp39-abi3-manylinux_2_28_aarch64.whl $CONT:/tmp/
docker cp $FW/flashinfer_jit_cache_sm121a-0.7.1+cu130-cp39-abi3-manylinux_2_28_aarch64.whl $CONT:/tmp/
docker cp $PATCH_DIR/. $CONT:/opt/dsv41-patches/

# --- 4.2 装 flashinfer 0.7.1（顺序关键！---
docker exec $CONT bash -c '
set -e
cd /tmp && tar xzf fi.tar.gz
pip install --no-deps --no-build-isolation ./flashinfer-0.7.1
pip install --no-deps ./flashinfer_cubin-0.7.1-py3-none-any.whl
pip uninstall -y flashinfer-jit-cache          # 旧 0.6.18 元数据必须卸掉（版本校验会拦）
pip install --no-deps --force-reinstall /tmp/flashinfer_jit_cache-0.7.1+cu130-cp39-abi3-manylinux_2_28_aarch64.whl
pip install --no-deps --force-reinstall /tmp/flashinfer_jit_cache_sm121a-0.7.1+cu130-cp39-abi3-manylinux_2_28_aarch64.whl
'

# --- 4.3 修复 1：AOT fp4_quantization_120f（关键！）---
# 上游 fp4_quantization.py:288-294 在 CUDA>=12.9 时把 SM12x 的 backend 统一改写成
# "120f"，但 sm121a wheel 只发 "121" ⇒ AOT miss ⇒ 回落源码编译 ⇒ 缺头文件就崩。
# 修复 = 补目录（.so 必须同步改名）+ manifest 加条目：
docker exec $CONT bash -c '
set -e
JIT=/opt/sglang/lib/python3.12/site-packages/flashinfer_jit_cache/providers/sm121a
mkdir -p $JIT/jit_cache/fp4_quantization_120f
cp $JIT/jit_cache/fp4_quantization_121/fp4_quantization_121.so \
   $JIT/jit_cache/fp4_quantization_120f/fp4_quantization_120f.so
python3 - <<PY
import json
p = "$JIT/manifest.json"
m = json.load(open(p))
if "fp4_quantization_120f" not in m["modules"]:
    m["modules"].append("fp4_quantization_120f"); m["modules"].sort()
    json.dump(m, open(p, "w"), indent=2)
print("  120f in manifest:", "fp4_quantization_120f" in m["modules"])
PY
'

# --- 4.4 修复 2：3rdparty 头文件补齐（若 BASE 也缺则跳过；用 0.6.18 的数据）---
# git-archive tarball 不含 3rdparty 子模块 ⇒ flashinfer/data/{cutlass,cccl,spdlog} 为空
# ⇒ JIT 编译报 cutlass/arch/barrier.h: No such file or directory。
# 若你的 BASE 是 0.6.18 且 3rdparty 完整（本集群是），从 BASE 补：
docker exec $CONT bash -c '
D=/opt/sglang/lib/python3.12/site-packages/flashinfer/data
for f in cutlass/include/cutlass/arch/barrier.h \
         cccl/libcudacxx/include/cuda/std/version \
         spdlog/include/spdlog/spdlog.h; do
  [ -e "$D/$f" ] && echo "  OK   $f" || echo "  MISS $f"
done
'
# 若上面有 MISS：docker cp 从旧 0.6.18 镜像取（见 §5 注意事项 1 的恢复命令）

# --- 4.5 apply 6 个 patch ---
docker exec $CONT bash -c 'cd /opt/dsv41-patches && bash apply.sh 2>&1 | tail -4'
# 期望：DONE applied=6 skipped=0（或先 applied=5 再单独 patch M7）
# 应用后 flash_mla_sm120.py 应为 f38e9b7c47e22ddb（M2+M7 叠加）

# --- 4.6 校验 ---
docker exec $CONT python3 -c "
import flashinfer; assert flashinfer.__version__ == '0.7.1', flashinfer.__version__
import py_compile
py_compile.compile('/sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py', doraise=True)
from sglang.kernels.ops.attention import flash_mla_sm120 as m
assert m._flashinfer_supports_dsv41() is True
print('  build checks OK')
"

# --- 4.7 commit（entrypoint 必须显式修正！）---
docker commit --change 'ENTRYPOINT ["python3", "-u", "/opt/dsv41/boot.py"]' \
              --change 'CMD ["run"]' $CONT $NEW
docker rm -f $CONT

# entrypoint 自检（sleep infinity 会被固化，必须确认）
docker image inspect $NEW --format 'EP={{.Config.Entrypoint}} CMD={{.Config.Cmd}}'
# 期望：EP=[python3 -u /opt/dsv41/boot.py] CMD=[run]
```

---

## 5. 阶段 D：镜像门禁（**不可跳过**——第 2 次部署失败的直接教训）

门禁必须**触发真实 kernel 编译与执行**，纯 version/import/md5 会被第 4 节的缺陷穿透：

```bash
#!/bin/bash
# verify_image.sh —— 四项门禁
IMG=dsv41-v41fp4:20261015
CONT=v41gate
docker rm -f $CONT >/dev/null 2>&1
docker run -d --name $CONT --gpus all --entrypoint sleep $IMG infinity

echo "=== 门禁 1：JIT 编译能力（清缓存后触发 fp4 量化 op）==="
docker exec $CONT bash -c 'rm -rf /root/.cache/flashinfer'
docker exec $CONT python3 -c "
import torch
q = torch.randn(4, 256, dtype=torch.bfloat16, device='cuda')
from flashinfer.quantization.fp4_quantization import fp4_quantize
try:
    fp4_quantize(q)   # 参数错没关系——走到 kernel 内部即证明 AOT/JIT 加载成功
    print('  JIT gate: loaded+ran')
except RuntimeError as e:
    assert 'globalScale' in str(e), e   # 已知 API 参数错 = 模块已加载
    print('  JIT gate: OK (module loaded; API-arg error expected)')
except Exception as e:
    raise SystemExit(f'  JIT gate FAIL: {e}')
"

echo "=== 门禁 2：上游测试 212 项 ==="
docker exec $CONT bash -c 'cd /tmp && tar xzf fi.tar.gz 2>/dev/null; cd /tmp/flashinfer-0.7.1 && \
  python3 -m pytest tests/mla/test_cake_dsv4.py -x -q 2>&1 | tail -2'
# 期望：212 passed

echo "=== 门禁 3：V41 端到端数值（sglang 正式入口）==="
docker cp e2e_v41_verify.py $CONT:/tmp/    # 见 §8 脚本清单
docker exec $CONT python3 /tmp/e2e_v41_verify.py 2>&1 | grep -E 'cosine|MATCH'
# 期望：cosine sim ≈ 0.999696，MATCH

echo "=== 门禁 4：entrypoint ==="
docker image inspect $IMG --format 'EP={{.Config.Entrypoint}} CMD={{.Config.Cmd}}'
# 期望：EP=[python3 -u /opt/dsv41/boot.py] CMD=[run]

docker rm -f $CONT
```

---

## 6. 阶段 E：部署窗口

### 6.1 停机
```bash
su - <user> -c "export WORKER_PASS=<worker密码> && cd $HOME/dsv41-4x-spark && ./start-tp4.sh stop"
```
**★ 停机会清掉 NFS 容器**。重启前必须手动恢复（用已有镜像，勿重建）：
```bash
sudo docker rm -f dsv41-nfs
sudo docker run -d --name dsv41-nfs --restart unless-stopped --privileged --network host \
  -v $HOME/models/DeepSeek-V4.1-Flash:/export:ro \
  -e "NFS_CLIENTS=WORKER_NETS" \
  dsv41-nfs:local
# 等 2049 监听（ss -tl | grep 2049）再继续
```
（`NFS_CLIENTS` 是**逗号分隔**——空格分隔会导致 exportfs netmask 报错。）

### 6.2 配置（head 节点宿主机）
```bash
D=$HOME/dsv41-4x-spark
cp -p $D/.env.tp4 $D/.env.tp4.bak-$(date +%Y%m%d-%H%M%S)   # 回滚锚点
sed -i 's|^IMAGE=.*|IMAGE=dsv41-v41fp4:20261015|' $D/.env.tp4
sed -i 's|^MAX_TOTAL_TOKENS=.*|MAX_TOTAL_TOKENS=13200000|' $D/.env.tp4
# EXTRA_CONTAINER_ENV 追加四个变量（enable-fp4kv.sh 自动做；或手动编辑）：
#   SGLANG_DSV4_KV_LAYOUT=v41
#   SGLANG_DSV4_COMPRESSED_KV_LAYOUT=fp4
#   SGLANG_SM120_FLASHMLA_BACKEND=flashinfer
#   DSV41_TRITON_SINGLE_CONFIG=1
```
> `MAX_TOTAL_TOKENS` 不提会被硬截断到 8M（KVCacheConfigurator 取 min(profiled, user_limit)）。
> `DSV41_TRITON_SINGLE_CONFIG=1` 把 Triton autotuner 缩成单 config，防其它 Triton 路径
> （autotune 的 do_bench 含 sync）作废 CUDA graph capture。

### 6.3 启动与验收
```bash
su - <user> -c "export WORKER_PASS=<密码> && cd $HOME/dsv41-4x-spark && ./start-tp4.sh serve"
# 构建期 ~7 分钟；launcher 可能报 exit 1（worker SSH 超时误报）——以下列两条为准：
docker ps --format '{{.Names}} {{.Status}}'        # 四容器 healthy
curl -s http://HEAD_IP:8888/health           # 200
docker logs dsv41-head 2>&1 | grep -oE 'bytes_per_full_token=[0-9.]+'    # 期望 912.25
docker logs dsv41-head 2>&1 | grep -oE 'full_token=[0-9]+'               # 期望 ~12.7M
```

### 6.4 冒烟
```bash
./start-tp4.sh smoke        # 期望 19+23=42
```

---

## 7. 阶段 F：门禁验收

| 门禁 | 命令 | 期望 |
|---|---|---|
| T3 缓存命中 | `python3 dsv41_long_cache.py`（宿主机，需引擎日志路径） | #2/#3 ≥85% |
| T4 qeval | 容器内 `qeval.py run <tag> --url http://HEAD_IP:8888/v1/chat/completions` | ≥71/75 |
| T4 batchtest | 容器内 `python3 /opt/dsv41-qa/batchtest.py <tag>` | 0 garbled |
| T5 soak | 8 并发长输出（脚本见 patches/gates_final.sh） | 0 garbled |
| T6 step | 8 轮单流 median（脚本见 patches/t6_retest.sh） | <33.64 ms |
| T4 paired | `qeval.py compare <V4基线>.json <V41>.json` | FAIL 题归因清楚 |

**测量注意**：性能测量前确认 `CUDA_LAUNCH_BLOCKING` 未设（拖慢 ~12%）；
`sleep_on_idle` 时日志 throughput 是伪影（0.0x tok/s），勿据此判断性能。

---

## 8. 随本文的脚本/patch 清单

| 文件 | 用途 |
|---|---|
| `patches/M1-layout-gate.patch` … `M7-flashinfer-dsv41.patch` | 6 个 sglang patch |
| `patches/apply.sh` | 依赖序应用 + md5 断言 + 幂等 |
| `patches/e2e_v41_verify.py` | 门禁 3：V41 端到端数值对比（cosine≥0.999） |
| `patches/verify_p0_v2.sh` | Triton capture 修复验证（P0） |
| `patches/start_nfs_v3.sh` | NFS 手动启动 |
| `patches/t6_retest.sh` / `gates_final.sh` | T6/T3/T4/T5 门禁脚本 |
| `patches/build_v41fp4.sh`（§4 即其内容） | 镜像构建 |
| `state-tp4/fi071-wheels/`（head 宿主机） | flashinfer 四件套持久化副本 + 重建说明 |

## 9. 回滚

```bash
cd $HOME/dsv41-4x-spark
cp -p .env.tp4 .env.tp4.bak-v41-$(date +%Y%m%d-%H%M%S)
cp -p .env.tp4.bak-<部署前备份> .env.tp4        # IMAGE=canary-roce + MAX_TOTAL_TOKENS=8000000 + 无 FP4 变量
su - <user> -c "... ./start-tp4.sh stop"
# 起 NFS（同 §6.1）
su - <user> -c "... ./start-tp4.sh serve"
```

---

## 10. 踩坑清单（每条都真实发生过）

1. **git-archive tarball 不含 3rdparty 子模块** ⇒ `flashinfer/data/{cutlass,cccl,spdlog}` 为空 ⇒
   JIT 编译 `fatal error: cutlass/arch/barrier.h`。必须验证头文件在位或从 0.6.18 补。
2. **AOT 命名 gap**：上游运行时把 SM12x 统一改写 `120f`（CUDA≥12.9 启用 family 特性），
   但 sm121a wheel 只发 `121`。补目录+manifest 双双到位才命中（只建目录不够——
   查找规则是 `jit_cache_dir/<mod>/<mod>.so` 且 mod ∈ manifest.modules）。
3. **jit-cache 版本校验**：旧 0.6.18 的通用元数据包不卸载会报
   `version (0.6.18) does not match (0.7.1)`。
4. **commit 固化 entrypoint**：`--entrypoint sleep` 起的构建容器 commit 后服务必挂
   （`sleep: invalid time interval 'run'`）。用 `docker commit --change` 或 Dockerfile 重写。
5. **stop 清 NFS**：每次重启前检查 `dsv41-nfs`；clients 必须**逗号**分隔。
6. **`MAX_TOTAL_TOKENS` 硬截断**：不提额则容量增益 +0%。
7. **`CUDA_LAUNCH_BLOCKING=1` 拖慢 ~12%**：性能数据必须在无诊断开关下采。
8. **launcher exit code 误报**：以 `docker ps` + `/health` + 引擎日志为准。
9. **PyPI 不可达时走 GitHub releases**；四包（python/cubin/jit-cache/sm121a）缺一不可。
10. **纯 import/md5 门禁不够**：必须触发真实 kernel（JIT gate + 上游测试 + 端到端数值）。

---

## 11. 参考文档（同目录）

| 文档 | 内容 |
|---|---|
| `PRODUCTION-DEPLOY-V41FP4.md` | 生产现状：配置、门禁结果、运维手册 |
| `FLASHINFER-V41-FEASIBILITY.md` | 为什么选升级而非改造（23KB 评估 + 8 轮探针） |
| `FLASHINFER-UPGRADE-PATH.md` / `FI071-INTEGRATION-DONE.md` | 0.7.1 安装与 M7 接入细节 |
| `FI071-V41-NUMERIC.md` / `FI071-V41-API.md` | 数值验证数据与 API 签名 |
| `QUALITY-REPORT-V41.md` | 质量门报告（两道 FAIL 均为测量边界效应） |
| `V41-DEPLOY-ATTEMPT3.md` / `V41-DEPLOY-SUCCESS.md` | 部署过程流水与成功摘要 |
| `WINDOW-RUNBOOK.md` | 停机窗口完整 runbook（含预飞审计） |
| `INCIDENT-2026-10-14.md` | 显存事故复盘（隔离测试 OOM 规范的由来） |
 
