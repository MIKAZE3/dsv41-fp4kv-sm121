#!/usr/bin/env python3
"""从 GitHub 拉回已发布文件，做敏感信息最终抽查。"""
import base64
import json
import re
import sys
import urllib.request

TOKEN = sys.argv[1]
REPO = "MIKAZE3/dsv41-fp4kv-sm121"
FILES = [
    "README.md",
    "docs/REPRODUCIBLE-DEPLOY-GUIDE.md",
    "docs/PRODUCTION-DEPLOY-V41FP4.md",
    "docs/V41-DEPLOY-SUCCESS.md",
    "docs/QUALITY-REPORT-V41.md",
    "docs/FLASHINFER-V41-FEASIBILITY.md",
    "docs/FLASHINFER-UPGRADE-PATH.md",
    "docs/FI071-INTEGRATION-DONE.md",
    "docs/V41-DEPLOY-ATTEMPT3.md",
    "patches/M1-layout-gate.patch",
    "patches/M2-gather-dequant.patch",
    "patches/M3-m4-pass-layout.patch",
    "patches/M5-triton-v41.patch",
    "patches/M6-pool-configurator.patch",
    "patches/M7-flashinfer-dsv41.patch",
    "patches/apply.sh",
    "scripts/e2e_v41_verify.py",
    "scripts/gates_final.sh",
    "scripts/start_nfs_v3.sh",
    "scripts/t6_retest.sh",
    "scripts/test_enable_apply.sh",
    "scripts/verify_p0_v2.sh",
]
PATTERNS = [
    r"192\.168\.\d+\.\d+",
    r"10\.0\.\d+\.\d+",
    r"dgx-work1",
    r"\byztai\b",
    r"yztai123456",
    r"W1N1",
    r"MIKAZE",  # 仓库 owner 名出现在 URL 之外不应出现
]

bad = 0
for f in FILES:
    url = f"https://api.github.com/repos/{REPO}/contents/{f}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"token {TOKEN}",
        "Accept": "application/vnd.github.raw+json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        content = r.read().decode("utf-8", "replace")
    for pat in PATTERNS:
        n = len(re.findall(pat, content))
        if n:
            print(f"  FAIL {f}: /{pat}/ x{n}")
            bad += n

print(f"TOTAL RESIDUAL: {bad}")
sys.exit(1 if bad else 0)
