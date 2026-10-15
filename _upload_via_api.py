#!/usr/bin/env python3
"""用 GitHub Contents API 上传整个目录（绕开 git https 协议问题）。"""
import base64
import json
import mimetypes
import os
import sys
import urllib.request

TOKEN = sys.argv[1]
REPO = sys.argv[2]          # e.g. MIKAZE3/dsv41-fp4kv-sm121
LOCAL = sys.argv[3]         # staging dir
BRANCH = "main"

API = "https://api.github.com"


def api(method, path, payload=None):
    req = urllib.request.Request(
        API + path, method=method,
        data=json.dumps(payload).encode() if payload else None,
        headers={
            "Authorization": f"token {TOKEN}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        })
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def put_file(path, content_bytes, message):
    url = f"{API}/repos/{REPO}/contents/{path}"
    # 检查是否已存在（需要 sha）
    sha = None
    try:
        req = urllib.request.Request(url, headers={
            "Authorization": f"token {TOKEN}",
            "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            sha = json.loads(r.read().decode())["sha"]
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    body = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode(),
        "branch": BRANCH,
    }
    if sha:
        body["sha"] = sha
    req = urllib.request.Request(url, method="PUT",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"token {TOKEN}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


# 收集文件
files = []
for root, dirs, names in os.walk(LOCAL):
    dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
    for n in sorted(names):
        full = os.path.join(root, n)
        rel = os.path.relpath(full, LOCAL).replace("\\", "/")
        files.append((rel, full))

print(f"uploading {len(files)} files to {REPO}@{BRANCH}")
for rel, full in files:
    with open(full, "rb") as f:
        data = f.read()
    msg = f"add {rel}"
    put_file(rel, data, msg)
    print(f"  OK {rel} ({len(data)} B)")

print("ALL-UPLOADED")
