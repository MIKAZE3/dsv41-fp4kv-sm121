#!/bin/bash
# enable-fp4kv.sh apply byte-level check (round 39)
set -uo pipefail
WORK=/tmp/enable_test
SRC=$HOME/dsv41-4x-spark/.env.tp4
ENABLE=/tmp/enable-fp4kv.sh

echo "=== 0. prepare copy ==="
rm -rf "$WORK"; mkdir -p "$WORK"; cp -p "$SRC" "$WORK/.env.tp4"
echo "  copy ready"

echo
echo "=== 1. original sha256 ==="
ORIG_SHA=$(sha256sum "$WORK/.env.tp4" | awk "{print \$1}")
echo "  $ORIG_SHA"
echo "  (production should be f81f5695e53cf99a...)"

echo
echo "=== 2. real apply (not dry-run) ==="
ENVF="$WORK/.env.tp4" bash "$ENABLE" 2>&1 | tail -14

echo
echo "=== 3. verify applied content ==="
echo "  --- three KV vars ---"
grep -oE "SGLANG_DSV4_KV_LAYOUT=v41|SGLANG_DSV4_COMPRESSED_KV_LAYOUT=fp4|SGLANG_SM120_FLASHMLA_BACKEND=triton" "$WORK/.env.tp4" | sed "s/^/    /"
echo "  --- MAX_TOTAL_TOKENS ---"
grep -E "^MAX_TOTAL_TOKENS=" "$WORK/.env.tp4" | sed "s/^/    /"
echo "  --- var occurrence count (each should be 1 = in-place replace) ---"
for v in SGLANG_DSV4_KV_LAYOUT SGLANG_DSV4_COMPRESSED_KV_LAYOUT SGLANG_SM120_FLASHMLA_BACKEND MAX_TOTAL_TOKENS; do
  printf "    %-38s %s\n" "$v" "$(grep -c "^$v=" "$WORK/.env.tp4")"
done
echo "  --- CRLF check (should be 0) ---"
printf "    CR=%s\n" "$(tr -cd "\r" < "$WORK/.env.tp4" | wc -c)"

echo
echo "=== 4. backup created? ==="
BAK=$(ls "$WORK"/.env.tp4.bak-fp4kv-* 2>/dev/null | head -1)
if [ -n "$BAK" ]; then ls -la "$BAK" | sed "s/^/  /"; else echo "  (no backup)"; fi

echo
echo "=== 5. backup == original? ==="
if [ -n "$BAK" ]; then
  B_SHA=$(sha256sum "$BAK" | awk "{print \$1}")
  echo "  backup sha256: $B_SHA"
  if [ "$B_SHA" = "$ORIG_SHA" ]; then echo "  OK backup==original"; else echo "  FAIL backup differs"; fi
else
  echo "  (no backup)"
fi

echo
echo "=== 6. --revert back to original ==="
ENVF="$WORK/.env.tp4" bash "$ENABLE" --revert 2>&1 | tail -6
AFTER_SHA=$(sha256sum "$WORK/.env.tp4" | awk "{print \$1}")
echo "  after-revert sha256: $AFTER_SHA"
if [ "$AFTER_SHA" = "$ORIG_SHA" ]; then echo "  OK revert byte-identical"; else echo "  FAIL revert differs"; fi

echo
echo "=== 7. cleanup ==="
rm -rf "$WORK"; echo cleaned
echo
echo "ENABLE-APPLY-TEST DONE"