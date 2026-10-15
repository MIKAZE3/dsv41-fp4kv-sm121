# FLASHINFER-V41-FEASIBILITY — flashinfer SM120 sparse-MLA kernel 改造支持 V41/V41_FP4 的可行性评估

> 本文取代同目录下早期的 `FLASHINFER-V41-FEASIBILITY.md` 初稿（2026-10-15 早版）。
> 早版只做了静态代码分析；本版加入了 8 轮隔离容器 GPU 实测 + 上游 PR 追查。

- 执行人：p22-flashinfer-v41 子代理
- 日期：2026-10-15
- 环境：node2 / `fi-probe` 隔离容器（`dsv41-fp4kv:20261014`，`--gpus all`，`torch.cuda.set_per_process_memory_fraction(0.15)`）
- 纪律：全程只读 + 隔离容器实验；**生产 `dsv41-worker` 未停未改**（`docker ps` 两次确认 Up）
- 原始探针：`p22-flashinfer-v41/probe{1,2,3,4,5,6,7,8}_*.py`，过程流水 `p22-flashinfer-v41/PROGRESS.md`

---

## 0. 三句话结论

1. **技术上完全可行，且工作量比预期小得多**——flashinfer 这个 kernel 是 **JIT `.cuh` 源码**（全包 0 个 `.so`），584 布局集中在 `model/kv_cache_traits.cuh` **一个 struct**，Python 层 584 硬编码只有 **5 处常量**。
2. **"布局伪装"是死路，已实测证伪**：把 V41 塞进 584 B 页 ⇒ 容量增益 **0%**（还是 584 B/token），输出误差 **28.7%**（stored-V 失真 65.6%）。
3. **但真正该做的不是改造，是升级**：**上游 flashinfer PR [#5983](https://github.com/flashinfer-ai/flashinfer/pull/5983) 已于 2026-10-03 合并**，标题就是 "SM120/SM121 DeepSeek-V4.1 mixed-cache sparse-MLA decode (528 B FP8 SWA + 288 B FP4 compressed)"，**点名 GB10/SM121**，37/37 benchmark 行赢过我们正在用的 584B `sparse` kernel，geomean **2.26×**。我们装的是 **0.6.18**，PR 在 **v0.7.1（10-09）** 里。

---

## Q1. flashinfer 这个 SM120 kernel 是什么形态？

### 结论：**纯 JIT 源码，不是预编译 `.so`。改 `.cuh` 即可，不需要重编 flashinfer 发行包。**

| 项 | 实测值 | 证据 |
|---|---|---|
| 版本 | **0.6.18** | `flashinfer.__version__` |
| 安装路径 | `/opt/sglang/lib/python3.12/site-packages/flashinfer` | `flashinfer.__file__` |
| `flashinfer/mla/` 内容 | **仅 3 个 .py**：`_core.py`(183,316 B)、`__init__.py`(1,261 B)、`_sparse_mla_sm120.py`(48,499 B) | `ls -la` |
| **`.so` 总数** | **0** | `find /opt/sglang/.../flashinfer -name "*.so" \| wc -l` → **0** |
| JIT 机制 | `from ..jit.mla import gen_sparse_mla_sm120_module` | `_sparse_mla_sm120.py` L60 |
| JIT 编译单元 | 5 个 `.cu`：`sparse_mla_sm120.cu` / `..._decode_dsv3_2.cu` / **`..._decode_dsv4.cu`** / `..._prefill.cu` / `..._jit_binding.cu` | `flashinfer/jit/mla.py:gen_sparse_mla_sm120_module()` |
| JIT 缓存 | `~/.cache/flashinfer/0.6.18` | 容器内实测（Oct 14 18:09 有产物） |

### kernel 本体文件清单（全部可读可改）

```
flashinfer/data/include/flashinfer/attention/sparse_mla_sm120/
├── decode_dsv4_kernel.cuh      43,697 B   ← ★ V4 decode warp-spec kernel 主体
├── decode_dsv3_2_kernel.cuh    30,935 B
├── prefill_kernel.cuh          77,806 B
├── model/
│   ├── kv_cache_traits.cuh     10,448 B   ← ★★ 584 布局的唯一权威定义
│   ├── model_type.h             2,498 B   ← ModelType / ScaleFormat / ComputeMode 枚举
│   └── scale_convert.cuh        3,005 B
├── common/
│   ├── kv_cache_io.cuh          5,449 B
│   ├── smem_layout.cuh         11,588 B
│   ├── xv_rope_mma.cuh          8,644 B
│   ├── q_rope.cuh               3,611 B
│   ├── fp8_quant.cuh            5,400 B
│   ├── online_softmax.cuh       3,217 B
│   ├── d2_load_b.cuh            4,552 B
│   └── scale_mma.cuh            4,567 B
└── arch/
    ├── cp_async.cuh  ldmatrix_sm120.cuh  mma_sm120.cuh
    └── barrier.cuh  common.cuh

flashinfer/data/csrc/sparse_mla_sm120_decode_dsv4.cu   10,211 B  ← 模板实例化入口
```

> ⚠️ **运维陷阱**（来自上游 PR #3817 的经验，我们在容器里已确认 `jit_cache` 存在）：
> 若镜像里带了 `flashinfer_jit_cache` 的 AOT `.so`，**改 `.cuh` 不会生效**——JIT loader 先找到 AOT 缓存就直接加载。
> 必须 mask/删除 `~/.cache/flashinfer/0.6.18/sparse_mla_sm120/` 强制重编。
> 我们容器里 `find ... -name "*.so"` = 0 ⇒ 当前镜像是纯 JIT，改 `.cuh` 立即生效（好事）。

---

## Q2. 584 → 528 的改造范围到底多大？

### 2.1 Python 层（`flashinfer/mla/_sparse_mla_sm120.py`）— 5 处常量 + 2 处 dispatch

| # | 行号 | 现状 | 需要改成什么 | 难度 |
|---|---|---|---|---|
| P1 | L141 | `_BPT_DSV4 = 584` | 加 `_BPT_DSV41 = 528`；`_bytes_per_token_for_model_type` 加分支（L183-187） | 常量替换 |
| P2 | L140 | `_BPT_DSV3_2 = 656` | 不动 | — |
| P3 | L190-224 | `_packed_kv_page_block_size()` 整除/last-dim 校验 | 从 `model_type` 推导 bpt（已支持），无需改结构 | **0 行** |
| P4 | L246-263 | `_decode_dsv4_dispatchable()` 要求 `d_qk == 512` | V41 也是 d_qk=512 ⇒ **不用改** | **0 行** |
| P5 | L78-106 / L107 | `_DECODE_DSV4_DISPATCH`(25 项) / `_DECODE_DSV4_PAGE_BLOCK_SIZE = 64` | V41 的 (nh, topk) 集合相同 ⇒ **不用加条目**（V41 tile=32 vs V4 tile=64 是 scale 粒度，不是 dispatch 维度） | **0 行** |
| P6 | L136-138 | `_MODEL_TYPE_DSV4 = 1` | 加 `_MODEL_TYPE_DSV41 = 3`，并贯通到 `model_type.h` 枚举 | 常量替换 |
| P7 | L329-368 | decode 分流 `model_type == DSV4 and kv_pbs == 64` | 加 V41 分支 | 10-15 行 |

**⇒ Python 层合计 ~30-50 行，纯常量替换 + 分支。**

### 2.2 kernel 层 — 一个 struct + 6 个派生 constexpr

**所有 584 布局知识集中在 `model/kv_cache_traits.cuh` L95-141 的 `KVCacheTraits<ModelType::DSV4>` 一处：**

```cpp
// L98-101   维度
D_NOPE = 448;  D_ROPE = 64;  D_QK = 512;  D_V = 512;
// L104-106  量化
QUANT_TILE = 64;  NUM_SCALES = 7;  SCALE_FORMAT = UE8M0_BYTE;
// L113-114  footer
SCALE_INLINE = false;  SCALE_BYTES_PER_TOKEN = 8;
// L119-121  ★ gmem 布局
KV_GMEM_STRIDE        = D_NOPE + D_ROPE*2 + 8 = 584;
KV_ROPE_GMEM_OFFSET   = 448;      // rope @ 448, 128 B bf16
KV_SCALE_GMEM_OFFSET  = 576;      // scale footer @ 576
// L126-128  smem 布局
KV_SMEM_STRIDE = 464;  KV_SMEM_COPY_BYTES = 448;  SCALE_IN_KV_SMEM = false;
// L132, L137
Q_NOPE_STRIDE = 464;  V_HAS_ROPE = true;
```

`decode_dsv4_kernel.cuh` 的派生常量（全部由上面 struct 推导，**无散落硬编码**）：

| 行号 | 常量 | V4 值 | V41 目标值 |
|---|---|---|---|
| L148 | `IO_STRIDE = D_NOPE + D_ROPE_C*2` | 576 | **512**（无独立 rope 区） |
| L194 | `N_V_CHUNKS = D_NOPE / V_CHUNK` | 7 | 16（tile 32） |
| L236 | `DSV4_BULK_NOPE_BYTES` | 448 | **512**（一次 bulk copy 覆盖 nope+rope） |
| L278 | scale footer 寻址 `section_pbs*IO_STRIDE + local*8` | footer@576, 8B | footer@512, **16 B** |
| L306 | rope 单独 `cp_async_bulk_g2s(data_base+D_NOPE, 128B)` | 独立 128 B 拷贝 | **删除这一条 bulk copy** |
| L765 | QK rope `d0 = D_NOPE + rope_dim_base + ...` | rope 在 q 的 [448:512) | q 布局不变（q 仍是 448 nope + 64 rope），**kernel 侧要把 V 的 rope 从"data 区 [448:512) 读"改为"与 nope 同区读"** |

### 2.3 V41 的 rope 在 data 区 ⇒ 偏移算法差异（核心难点）

V4 的 V 向量 = `nope[0:448]`(fp8×ue8m0) ⊕ `rope[448:512]`(bf16，无 scale)
V41 的 V 向量 = `data[0:512]`(全 fp8×ue8m0，tile 32)

**这不是"布局平移"，是三处结构性变化：**

| 变化 | 影响 | 难度 |
|---|---|---|
| ① `D_NOPE` 448→512（V 的 512 维全是 fp8 nope） | `N_V_CHUNKS` 7→16；`NUM_SCALES` 7→16；XV MMA 的 `acc_nope` 维度从 448 pad 到 512 | 中 |
| ② rope 从"bf16 独立区 + 独立 bulk copy + CUDA-core 标量 FMA"（`V_HAS_ROPE=true` 路径，L306/L765）塌缩为"fp8 同区" | **必须删掉整条 rope 特化路径**，V = 单区 512 fp8 | 中（是删除而非新增，反而简化） |
| ③ scale tile 64→32、footer 8B→16B、`SCALE_BYTES_PER_TOKEN` 8→16 | `KV_SCALE_GMEM_OFFSET` 576→512；smem scale buffer 大小翻倍 | 低 |
| ④ `KV_SMEM_STRIDE` 464→528，需重新验证 16B 对齐 + bank conflict | 464/4=116, 116%32=20（clean）；**528/4=132, 132%32=4 ⇒ 4-way bank conflict**（`kv_cache_traits.cuh` L69 注释说 DSV3_2 的 528 "acceptable"） | 低（已知可接受） |

**⇒ kernel 层 = 1 个新 `KVCacheTraits<DSV41>` struct（~45 行）+ 删 rope 特化分支（~40 行删除）+ 6 个 constexpr 参数化。估算 150-250 行改动，且**多数是删除**。**

### 2.4 warp-spec tile / shared memory 是否依赖 584？

**不依赖，且已实测验证。** `decode_dsv4_kernel.cuh` L194-211 的 smem 预算全部由 `DSV4_BI`(=64) 和 `HPB`(=16) 驱动，与 bytes_per_token 无关；`_BI = 64` 是 **candidates 的 tile 尺寸**，不是字节。V41 的 `page_block_size` 仍是 64（V41 的 `page_align=512` 只影响 page 级 padding，`page_bytes(64)=33792`）。

**⇒ tile 形状、smem 布局与 584 解耦，只需重算 `KV_SMEM_STRIDE` 一处。**

### 2.5 `_DECODE_DSV4_PAGE_BLOCK_SIZE` / `_DECODE_DSV4_DISPATCH` 是否要加 V41 条目？

**不要。** 实测：`_decode_dsv4_dispatchable(16, 16, 512, 512, 64)` → `True`。V41 的 `d_qk` 仍是 512、`page_block_size` 仍是 64、`(num_heads, topk)` 集合不变。**只需在 `_paged_attention` 的分流条件里加一个 `model_type == DSV41` 分支。**

### 2.6 ★ 但真正的墙不在 flashinfer，在我们自己的 sglang 侧

`flash_mla_sm120.py`（我们的 M5 patch）有**两处显式硬门**：

```python
# L349-357（prefill 路径）和 L752-759（decode 路径）
if k_cache.shape[-1] != 584:
    raise ValueError("the flashinfer SM120 backend reads only the 584-byte V4 KV layout, ...")
```

以及 L63-68 的自认：
> "The FlashInfer backend's page-split is written for the 584-byte V4 page and fails loud on anything else, so it must not advertise V4.1 support"

还有 page-split 的 4 组常量（L532-540, L716-719, L726-729）：
`_PBS_DST=64`、`_BYTES_PER_DST_PAGE_PADDED=37440`、`DATA_PER_SUB=36864`、`SCALE_PER_SUB=512`、`bpt=584`。

**⇒ 即使改了 flashinfer，也还要改我们自己的 ~5 处常量 + 拆掉 2 个 584 断言。这些是我们自己写的守卫，拆起来没有兼容性风险。**

---

## Q3. 三条"巧劲"路径的可行性判定

### A. 布局伪装 —— ❌ **实测证伪，双输**

**做法**：把 V41 的 528 B 页填成 584 B，让 flashinfer 以为它是 V4 页。

**容量账（先算账，已实测）：**

| 方案 | B/token | vs V4 容量增益 |
|---|---|---|
| V4（现状生产） | 584 | 基准 |
| V41 | 528 | **+9.6%** |
| V41_FP4 | 288 | **+50.7%** |
| **伪装成 584 B 页** | **584** | **+0.0%** |

**⇒ 第一重死因：伪装页还是 584 B，容量一点不省。** 用户问的"容量增益从 +87% 降到多少"——注意 +87% 这个数不对（那是 584/288 相关的错误口径）。按 `r4-capacity.md` 的精算，本集群引擎口径下"V41 主 + FP4 副"真实增益是 **+30.3%**（bpft 1670.75→1281.91）。而伪装路径：**+0%**。

**第二重死因：数值实测不过关。**

`probe7_clean.py` 隔离容器实测（GB10 sm121，B=8/NH=16/topk=512/pbs=64）：

| 组 | out rel err | 说明 |
|---|---|---|
| **D0** 真 V4 数据 | **3.114e-02** | flashinfer fp8+bf16 MMA 固有误差基线 |
| **D1** V41 伪装进 584 B 页 | **3.329e-02**（vs 自身存的 V） | kernel 没引入额外误差 ✓ |
| **D1 vs 真 V41** | **2.871e-01** | **输出误差 28.7%** |
| stored-V 保真度 | **6.562e-01** | **存的 V 与真 V41 V 差 65.6%** |

**为什么必然失真（结构性，非实现 bug）：**

V4 页只有 **448 B fp8 nope + 128 B bf16 rope + 8 B scale footer**；
V41 需要 **512 B fp8 + 16 B scale**。

- V41 的 16 个 scale（每 32 值一个）**在 V4 的 7 个 scale byte 里放不下**。只能把相邻两个 32-tile 合并成 64-tile 取 min scale ⇒ nope 量化粒度从 32 退到 64。
- V41 的 rope（data[448:512]）本来也是 fp8+scale，塞进 V4 的 bf16 rope 区要先反量化再转 bf16，**而 V4 的 scale footer 不作用于 rope**（`probe6` 实测：scale byte=0 时 nope 精确 0 但 rope 仍精确 0.25）⇒ rope 的 scale 信息**无处安放**，只能烧进 bf16 值里。
- 用户提的"把 V41 512 NOPE 的前 448 当 V4 NOPE、后 64 B + rope 区放量化后的 rope"——这正是 D1 的做法。**前 448 的 scale 粒度掉一半；rope 的 scale 丢失。**

**⇒ 结论 A：不可行。容量 +0%，精度 28.7% 输出误差。已用隔离容器实测证伪，不需要再投入。**

### B. 只用 V41_FP4 作 extra，main 仍 V4 —— ❌ **代码级不可能**

三层独立阻断，任一层都足够判死：

1. **`is_valid_kv_layout_pair()`（`kv_layout.py` L71-74）**：
   ```python
   return extra_kv is kv or (kv is KVLayout.V41 and extra_kv is KVLayout.V41_FP4)
   ```
   ⇒ `(V4, V41_FP4)` **直接 False**。要么同布局，要么必须 V41 主。
2. **flashinfer C++ 层硬检查**（`probe2` 实测）：
   ```
   Check failed: kv.size(-1) == bpt (528 vs. 584) : extra_kv_cache last dim must be bytes_per_token=584, got 528
   Check failed: kv.size(-1) == bpt (288 vs. 584) : extra_kv_cache last dim must be bytes_per_token=584, got 288
   ```
   extra cache 的 bpt 与 main 同源（都走 `_bytes_per_token_for_model_type`）。
3. **`r4-capacity.md` R4-11/12 已证**：副缓存成本通过 `low_ratio` 项计价，主/副池**布局必须成对**。

**⇒ 结论 B：技术上不可能，且即使可能收益也有限（副池只占 bpft 的一部分）。**

### C. 混合 dispatch（main=flashinfer V4 + extra=Triton V41_FP4）—— ⚠️ **被上游 PR #5983 降级为"没必要"**

这条在 PR #5983 之前是最稳的路（层 68 已验证 LSE merge）。但：

- `is_valid_kv_layout_pair` 仍然要求 main=V41 才能配 V41_FP4 extra ⇒ **main 保持 V4 就吃不到 FP4 extra**，还是回到 B 的死路。
- 若 main 也切 V41，那就是 Q2 描述的完整改造，与直接升级上游等价。
- 上游 #5983 已经在**同一个 kernel 里同时读 528 B main + 288 B extra** 并做 LSE merge，且 37/37 赢过 584B sparse。

**⇒ 结论 C：技术上最稳但收益被 A/B 的阻断条件抵消；且上游已交付更好的东西。**

---

## Q4. 上游状态 —— ★★★ 已经做完了

| 项 | 事实 | 证据 |
|---|---|---|
| **官方 PR** | **[#5983](https://github.com/flashinfer-ai/flashinfer/pull/5983)** "feat(cake_dsv4_sparse_mla): SM120/SM121 DeepSeek-V4.1 mixed-cache sparse-MLA decode (528 B FP8 SWA + 288 B FP4 compressed, BF16 Q)" | GitHub API |
| 状态 | **merged 2026-10-03T02:19:08Z** | `merged_at` |
| 作者 | `yyihuang`（COLLABORATOR） | — |
| **点名 GB10/SM121** | perf 表有 **GB10** 一行；入口限定 `SM120/SM121 only` | PR body |
| 性能（vs 我们正在用的 584B `sparse` B1） | **GB10: 37/37 rows faster, geomean 2.26×, min 1.28×**；RTX PRO 6000: 2.46×；RTX 5090: 2.48× | PR body benchmark 表 |
| 数值 | L1（fp32 exact-dequant oracle on **same** quantized cache）out max\|err\| **0.00195**、LSE max\|err\| **3.8e-6**；L2 rms-rel ~2.5e-2（= 缓存量化误差，与 `sparse` bf16 路由**相同**） | PR body |
| 验证 | RTX 5090 (12.0a) **95 passed / 1 skipped**；compute-sanitizer synccheck+memcheck **0 errors**；CUDA 12.9 / 13.0 / 13.3 三套工具链交叉验证 | PR body |
| 写入侧 | 附带 528 B main-cache writers `dsv41_fp8_quantize_pack_sparse_mla_cache` / `..._append_...`；288 B FP4 writers 复用 | PR body |
| **默认路由不变** | `backend="auto"` 仍走手写 SM120 `sparse`（584B）⇒ **升级不破坏现有生产** | PR body |
| 关闭的 issue | closes SM120/SM121 part of **#5936**；tracker **#4254** | PR body |

**入口 API：**
```python
trtllm_batch_decode_sparse_mla_dsv4(..., backend="cake", kv_cache_format="fp8_dsv41_fp4_ca")
SparseMLASm120Wrapper(backend="cake", kv_scale_format="ue8m0_g32",
                      extra_kv_fp4=True, compute_precision="bf16")
```

**版本时间线：**

| 版本 | 发布日期 | 与 #5983 的关系 |
|---|---|---|
| **0.6.18（我们装的）** | — | PR 之前的**扁平单文件**结构：`flashinfer/mla/_sparse_mla_sm120.py` |
| **v0.7.1** | **2026-10-09** | **必然含 #5983**（10-03 merged）；且发布了 `flashinfer_jit_cache_sm121a-0.7.1+cu130/cu134` aarch64 wheel ⇒ **GB10/SM121 有预编译 AOT 缓存，省去首次 JIT 编译** |
| v0.7.2rc3 | 2026-10-13 | 最新 |

**0.7.x 结构变化（重要）**：`flashinfer/mla/_sparse_mla_sm120.py` → `flashinfer/mla/_sparse_mla_sm120/` **目录**，含
`_api.py`(71,811 B) / `_policy.py`(24,262 B) / `_prepared.py`(19,466 B) / `_execution.py` /
`_dsv4_nvfp4.py` / `_cake_dsv4_nvfp4.py` / `_cake_dsv4_nvfp4_rope_insert.py` / `_calibration.py` /
**`cake_dsv41_mixed.py`(59,750 B)** ← #5983 的主体。

**相关但未合并的 PR（供参考，说明"改造"方向上游也走过）：**
- [#5763](https://github.com/flashinfer-ai/flashinfer/pull/5763) "runtime page sizes for DSV4 NVFP4 attention" — closed **unmerged** 10-06；记录了 vLLM 侧 page-geometry 依赖（vllm-project/vllm#57292）和 DeepGEMM pin（#59385）
- [#3817](https://github.com/flashinfer-ai/flashinfer/pull/3817) topk=256 decode-dsv4 — closed unmerged；但留下关键运维警告：**AOT `jit_cache` 会 shadow 改过的源码，必须 mask 才能强制重编**
- [#3374](https://github.com/flashinfer-ai/flashinfer/pull/3374) SM120 sparse-MLA 本体 — merged 2026-06-17

**sglang 上游**：本地 `sgl-workspace/sglang` 已有 `flash_mla_sm120_supports_v41_kv_layouts()`（L63-68），
当前 `return _sm120_default_backend == "triton"`——即 sglang 侧**只认 Triton 后端支持 V41**。
升级 flashinfer 后需要把这个判断扩展为"flashinfer ≥0.7.1 且 `backend=cake`"。

---

## Q5. 工作量估算表

| 路径 | 改动文件数 | 大致行数 | 需重编 kernel？ | 隔离环境可验证？ | 预期性能（vs Triton 311 µs/层） | 判定 |
|---|---|---|---|---|---|---|
| **D. 升级 flashinfer 0.6.18 → 0.7.1+** | 1（`_sparse_mla_sm120.py` 换成包 + 我们的 5 个 patch 重新 apply） | 重 apply ~50-150 行 | **否**（有 `sm121a` AOT wheel） | **是**，完全隔离 | **上游实测 GB10 geomean 2.26× vs 584B sparse**；584B sparse 我们实测 34.5 µs/层（topk=512）⇒ 预估 V41 混合 ~15-20 µs/层，**比 Triton 311 µs 快 15-20×** | ★★★★★ |
| **E. 自己改 flashinfer `.cuh` 加 DSV41** | 4-5（`kv_cache_traits.cuh`、`decode_dsv4_kernel.cuh`、`model_type.h`、`_sparse_mla_sm120.py`、我们自己的 `flash_mla_sm120.py`） | 150-250 行（kernel）+ 30-50 行（py）+ 50 行（我们的守卫） | **是**（JIT，首次 ~分钟级） | **是** | 与 D 等价（本来就是同一个 kernel 家族） | ★★★☆☆ |
| A. 布局伪装 | 2-3 | 80-120 | 否 | **已实测** | **+0% 容量，28.7% 输出误差** | ❌ 已证伪 |
| B. V41_FP4 作 extra，main V4 | — | — | — | 代码级不可能（`is_valid_kv_layout_pair` + C++ `kv.size(-1)==bpt`） | — | ❌ |
| C. 混合 dispatch | 3-4 | 100-150 | 否 | 是 | main V4（快）+ extra Triton（慢）加权，收益被 B 的阻断抵消 | ⚠️ 不推荐 |

### D 路径的详细工作量（推荐）

| 步骤 | 工作量 | 风险 |
|---|---|---|
| 1. 隔离容器装 flashinfer 0.7.1 + cu130，跑 #5983 的 `test_cake_dsv41_mixed.py` 子集 | **0.5 天** | 低（上游 95 passed） |
| 2. 把我们已有的 5 个 patch 重新 apply 到 0.7.x 的新目录结构 | **0.5-1 天** | **中**——`_sparse_mla_sm120.py` → 包，import 路径全变 |
| 3. `flash_mla_sm120.py` L63-68 的 `flash_mla_sm120_supports_v41_kv_layouts()` 扩展 + 拆 2 处 584 断言 + page-split 常量参数化 | **1 天** | 低（都是我们自己的守卫） |
| 4. `pool_configurator.py` L989 `kv_bytes` 随布局走（`r4-capacity.md` R4-13 已证必须改，否则 +0%） | **0.5 天** | 低 |
| 5. bit-exact + 语义 oracle 回归（FP64 oracle / 10,543-token md5 `51b1b897…` / relay mismatch=0） | **1-2 天** | **中**——V41 把 RoPE 也量化成 fp8，是**数值语义变化**，必须过 oracle |
| 6. 四节点灰度 + 回滚演练 | **1 天** | 低（`dsv41-rollback:pre-fp4-final` 已备） |
| **合计** | **4-6 人天** | |

---

## 6. 推荐路线

> **不要改造 flashinfer——升级它。**
> **把 `flashinfer` 从 0.6.18 升到 ≥0.7.1，直接启用 PR [#5983](https://github.com/flashinfer-ai/flashinfer/pull/5983) 的 `backend="cake"` + `fp8_dsv41_fp4_ca` 路由：它正是 528 B FP8 main + 288 B FP4 extra 的 SM121 kernel，上游已在 GB10 上实测 37/37 行赢过我们正在用的 584 B kernel（geomean 2.26×），且默认路由不变、升级不破坏现有 V4 生产。**

**理由：**

1. **上游已经做完了，而且做得比我们自己改好。** PR #5983 的 L1 oracle 误差 0.00195 / LSE 3.8e-6、95 passed、compute-sanitizer 0 errors、三套 CUDA 工具链交叉验证——这是我们自己改 `.cuh` 要达到的门槛，而现在是免费的。
2. **"巧劲"全是死路，已用隔离容器实测证伪**：布局伪装容量 +0% 且输出误差 28.7%；V41_FP4 extra 被 `is_valid_kv_layout_pair` + C++ `kv.size(-1)==bpt` 双重阻断；混合 dispatch 被同一阻断条件抵消。
3. **我们自己改的门槛比看起来高**：584 布局虽然集中在一个 struct（Q2 已精确定位），但 V41 的 rope-in-data 是**结构性变化**（删整条 rope 特化路径 + `N_V_CHUNKS` 7→16 + scale tile 64→32），不是常量替换。上游用了 ~60 KB 的 `cake_dsv41_mixed.py` + 生成式 `csrc/cake_dsv4/sm_120a/`（9 个 head 数 × decode/dual-cache/split-merge）。
4. **唯一真正的新增工作是"接进来"**：重 apply 我们 5 个 patch、扩展 `supports_v41_kv_layouts()`、改 `pool_configurator.py:989` 的 `kv_bytes`（否则容量增益 **+0%**，见 `r4-capacity.md` R4-4）、过语义 oracle。**合计 4-6 人天。**

**立即行动（今天就能做，完全隔离）：**
```
docker run -d --name fi71 --gpus all --entrypoint sleep <base> infinity
pip install flashinfer==0.7.1  (或从 v0.7.1 release 拉 flashinfer_jit_cache_sm121a-0.7.1+cu130)
python -m pytest tests/mla/test_cake_dsv41_mixed.py -x -q   # 上游 95 passed
# 然后跑我们自己的 528B main + 288B extra 构造，与 Triton 输出对比 + 计 34.5 µs 基线
```

---

## 附录：关键实测数据存档

| 探针 | 测了什么 | 关键输出 |
|---|---|---|
| `probe1_pagesize.py` | page-size 硬校验 | 528 → `last dim must be 584`；`135168 % 584 = 264` → `not divisible`（与用户给的 264 一致） |
| `probe4_correct.py` | 页面布局 + 基线 | V4 页 = data 区 `[0:pbs*576)` + footer 区 `[pbs*576:pbs*584)`；**34.5 µs/layer** |
| `probe6_semantics.py` | kernel 语义标定 | nope fp8×ue8m0 **精确**（rel err 0.000e+00）；rope bf16 **精确**；scale=0 ⇒ nope 0、rope 仍 0.25（footer 只管 nope）；V=0 ⇒ out 0 无 NaN |
| `probe8_lse.py` | LSE +log2(topk) 偏置 | offset = log2(topk)（128→7.01, 256→8.00, 512→9.01, 1024→9.99）；`topk_length=1` ⇒ offset 0.0135 消失；`attn_sink=0` ⇒ out mean 0.875→0.8733 |
| `probe7_clean.py` | **伪装 D1 干净误差** | D0 3.114e-02 / D1-vs-自身 3.329e-02 / **D1-vs-真V41 2.871e-01** / stored-V 保真 6.562e-01 |
| `probe2_disguise.py` | extra cache 布局 | 528B/288B extra ⇒ `Check failed: kv.size(-1) == bpt` |

**集群状态：生产 `dsv41-worker` 全程 Up，未停未改。隔离容器 `fi-probe` 已创建（可用 `docker rm -f fi-probe` 清理）。**
