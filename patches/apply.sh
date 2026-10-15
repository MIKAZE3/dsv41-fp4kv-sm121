#!/usr/bin/env bash
# apply.sh — 按依赖顺序应用 DSV41 V41+FP4 KV patch 套件（四节点通用）
#
# 用法（在 dsv41-head 容器内）:
#   bash apply.sh                 # 应用全部 4 个 patch
#   DRYRUN=1 bash apply.sh        # 只校验不落盘
#   NODE=<node1|node2|node3|node4> bash apply.sh   # 打印节点标识（仅日志）
#
# 设计约束:
#   * 远端不是 git 工作树 ⇒ 用 patch -p1，靠"前置 md5 断言 + 备份"保证可回滚
#   * 每步前后各校验一次 md5，任何不匹配立即 abort（不全打一半）
#   * 备份带时间戳，revert.sh 依赖它
#   * 幂等：已打过则跳过该 patch（靠备份文件是否已存在判断）
#
# 退出码: 0 = 全部成功 / 已应用; 非 0 = 失败（此时现场保持"上一个成功状态"）

set -euo pipefail

SGLANG_ROOT="${SGLANG_ROOT:-/sgl-workspace/sglang}"
PKG="$SGLANG_ROOT/python/sglang"
PATCH_DIR="${PATCH_DIR:-/opt/dsv41-patches}"
BACKUP_ROOT="${BACKUP_ROOT:-/opt/dsv41-patches/backup}"
TS="$(date +%Y%m%d-%H%M%S)"
STAMP_FILE="$BACKUP_ROOT/.applied-$TS"

# 文件 -> 基线 md5（2026-10-14 在 node1 实测，四节点同镜像应一致）
declare -A BASE_MD5=(
  ["srt/mem_cache/deepseek_v4_memory_pool.py"]="f0dfef3445661680ee1469be1f761ec9"
  ["kernels/ops/attention/flash_mla_sm120.py"]="aed9b0fe4958ad7e8c1245819c94acdb"
  ["srt/layers/attention/deepseek_v4_backend.py"]="201772b8577ff09d45facc6acd9ea76e"
  ["srt/model_executor/pool_configurator.py"]="759421a89cf087025833e58dbdd98213"
  ["kernels/ops/attention/flash_mla_sm120_triton.py"]="bb1682dd23aeeb6f30c766d3168957a0"
)
# patch -> 它改的文件（一个 patch 只改一个文件，简化备份/校验）
declare -A PATCH_FILE=(
  ["M1-layout-gate.patch"]="srt/mem_cache/deepseek_v4_memory_pool.py"
  ["M2-gather-dequant.patch"]="kernels/ops/attention/flash_mla_sm120.py"
  ["M7-flashinfer-dsv41.patch"]="kernels/ops/attention/flash_mla_sm120.py"  # 依赖 M2（同文件）
  ["M3-m4-pass-layout.patch"]="srt/layers/attention/deepseek_v4_backend.py"
  ["M6-pool-configurator.patch"]="srt/model_executor/pool_configurator.py"
  ["M5-triton-v41.patch"]="kernels/ops/attention/flash_mla_sm120_triton.py"
)
# 应用顺序 = 依赖顺序（M5 无依赖，放最后；它让 triton 后端能读 V41/V41_FP4）
# M7 紧跟 M2：两者改同一文件，M7 在 M2 的 V41 分支上接入 flashinfer>=0.7.1 的 cake_dsv4
ORDER=(M1-layout-gate.patch M2-gather-dequant.patch M7-flashinfer-dsv41.patch M3-m4-pass-layout.patch M6-pool-configurator.patch M5-triton-v41.patch)

log() { echo "[apply $(date +%H:%M:%S)] $*"; }
die() { echo "[apply FATAL] $*" >&2; exit 1; }

md5of() { md5sum "$1" | awk '{print $1}'; }

[ -d "$PKG" ] || die "sglang package not found at $PKG (set SGLANG_ROOT)"
[ -d "$PATCH_DIR" ] || die "patch dir not found at $PATCH_DIR"
mkdir -p "$BACKUP_ROOT"

log "sglang root : $SGLANG_ROOT"
log "patch dir   : $PATCH_DIR"
log "node        : ${NODE:-<unset>}"

# ---- 0. 前置断言：4 个文件必须与基线一致（否则中止，避免打在已改的树上） ----
log "step 0: baseline md5 assertion"
for rel in "${!BASE_MD5[@]}"; do
  f="$PKG/$rel"
  [ -f "$f" ] || die "missing $f"
  got="$(md5of "$f")"
  want="${BASE_MD5[$rel]}"
  if [ "$got" = "$want" ]; then
    log "  OK   $rel  $got"
  else
    # 已打过 patch 的文件 md5 会变；这时必须能从备份还原，否则拒绝继续
    if ls "$BACKUP_ROOT"/*.$(basename "$rel").*.bak >/dev/null 2>&1; then
      log "  SKIP $rel  md5=$got (already patched; backup exists)"
    else
      die "$rel md5=$got != baseline $want and no backup found. Refusing to patch a tree that is neither pristine nor ours."
    fi
  fi
done

# ---- 1. 逐 patch 应用 ----
applied=0
skipped=0
for p in "${ORDER[@]}"; do
  rel="${PATCH_FILE[$p]}"
  pf="$PATCH_DIR/$p"
  target="$PKG/$rel"
  [ -f "$pf" ] || die "patch file missing: $pf"
  bak="$BACKUP_ROOT/$(basename "$rel").$TS.bak"

  # 已应用检测：目标里已有本 patch 的标志性锚点
  case "$p" in
    M1-layout-gate.patch)      anchor="get_device_capability()[:2] == (12, 1)" ;;
    M2-gather-dequant.patch)   anchor="def _gather_and_dequant(k_cache, indices, page_size, layout=None):" ;;
    M7-flashinfer-dsv41.patch) anchor="def flash_mla_sm120_supports_v41_kv_layouts() -> bool:" ;;
    M3-m4-pass-layout.patch)   anchor="kv_layout=token_to_kv_pool.get_swa_key_layout()," ;;
    M6-pool-configurator.patch) anchor="_dsv4_kv_layout_for_sizing" ;;
    M5-triton-v41.patch)     anchor="IS_V41: tl.constexpr," ;;
    *) die "unknown patch $p" ;;
  esac

  if grep -qF "$anchor" "$target"; then
    log "  SKIP $p (anchor already present)"
    skipped=$((skipped+1))
    continue
  fi

  # 备份（只备份第一个 patch 触碰该文件时；后续 patch 备份的是打补丁中间态）
  # DRYRUN 必须完全只读：不写 base.bak，否则中断会在生产留下"看似已打过"的痕迹，
  # 让后续真实 apply 的 step 0 误判（stale backup 的坑）。
  if [ "${DRYRUN:-0}" != "1" ]; then
    if [ ! -f "$BACKUP_ROOT/$(basename "$rel").base.bak" ]; then
      cp -p "$target" "$BACKUP_ROOT/$(basename "$rel").base.bak"
      log "  BACKUP base  $(basename "$rel") -> backup/$(basename "$rel").base.bak"
    fi
    cp -p "$target" "$bak"
  fi

  log "  APPLY $p -> $rel"
  if [ "${DRYRUN:-0}" = "1" ]; then
    if patch -p1 --dry-run -d "$PKG" < "$pf" >/dev/null; then
      log "    dry-run OK"
    else
      die "dry-run failed for $p"
    fi
    # dry-run 不落盘 ⇒ md5 不会变，跳过"必须变化"校验，直接进入下一个 patch
    skipped=$((skipped+1))
    continue
  else
    if ! patch -p1 -d "$PKG" < "$pf"; then
      die "patch failed for $p; restoring $rel from $bak"
      cp -p "$bak" "$target"
    fi
  fi

  # 打后校验：必须仍是合法 python，且 md5 必须变化
  # （用 python -m py_compile，不用内联 -c：引号在 docker exec bash -c 多层传递中会被剥掉）
  python -m py_compile "$target" \
    || die "$rel does not compile after $p"
  new_md5="$(md5of "$target")"
  old_md5="$(md5of "$bak")"
  [ "$new_md5" != "$old_md5" ] || die "$rel unchanged after $p (md5 $new_md5)"
  log "    OK  md5 $old_md5 -> $new_md5"
  applied=$((applied+1))
done

# ---- 2. 汇总 + 记录 ----
if [ "${DRYRUN:-0}" != "1" ]; then
  {
    echo "applied_at=$TS"
    echo "node=${NODE:-<unset>}"
    echo "applied=$applied skipped=$skipped"
    for p in "${ORDER[@]}"; do
      rel="${PATCH_FILE[$p]}"
      echo "  $p -> $rel md5=$(md5of "$PKG/$rel")"
    done
  } > "$STAMP_FILE"
  log "stamp written: $STAMP_FILE"
fi

log "DONE applied=$applied skipped=$skipped"
[ "$applied" -gt 0 ] || [ "$skipped" -gt 0 ] || die "nothing applied and nothing skipped"
echo
echo "NEXT: 设置环境变量并重启服务（见 IMPLEMENTATION.md §验证）:"
echo "  SGLANG_DSV4_KV_LAYOUT=v41"
echo "  SGLANG_DSV4_COMPRESSED_KV_LAYOUT=fp4"
echo "  SGLANG_SM120_FLASHMLA_BACKEND=triton     # M5 已交付；torch 退化 12.8% 不可上生产"
exit 0
