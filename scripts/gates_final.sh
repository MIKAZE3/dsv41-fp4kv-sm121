#!/bin/bash
# T3 + T4 剩余 + T5
set -uo pipefail

echo "=== T3: 前缀缓存命中 ==="
timeout 400 python3 $HOME/dsv41-e2e-host/dsv41_long_cache.py 2>&1 | tail -5 | sed 's/^/  /'

echo
echo "=== T4: paired compare（qeval vs V4 基线）==="
docker exec dsv41-head bash -c "cd /state/baselines && python3 /opt/dsv41/scripts/qeval.py compare qeval-v4-baseline-r33.json qeval-qeval-v41-final.json 2>&1 | tail -14" | sed 's/^/  /'

echo
echo "=== T5: 并发 soak（缩量版：8 并发 × 60s）==="
docker exec dsv41-head python3 -c "
import json, time, urllib.request, threading, concurrent.futures

def one(i):
    req = urllib.request.Request('http://127.0.0.1:8888/v1/chat/completions',
        data=json.dumps({'model':'deepseek-v4.1-flash','temperature':0.7,'max_tokens':256,
        'chat_template_kwargs':{'thinking':False},
        'messages':[{'role':'user','content':f'Tell story {i} about a sea voyage, 200 words.'}]}).encode(),
        headers={'Content-Type':'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read().decode())
    return resp['usage']['completion_tokens'], time.time()-t0, resp['choices'][0]['message']['content']

t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(8) as ex:
    results = list(ex.map(one, range(8)))
wall = time.time() - t0
tot = sum(r[0] for r in results)
print(f'  8 concurrent: wall={wall:.1f}s aggregate={tot/wall:.1f} tok/s per-req={[f\"{r[0]/r[1]:.1f}\" for r in results]}')
# 崩坏检测
import re
def garbage(txt):
    cjk = re.search(r'[　-鿿가-힯]', txt) is not None
    body = [ch for ch in txt if not ch.isspace()]
    latin = sum(1 for ch in body if ch.isascii() and (ch.isalpha() or ch in \".,;:'\\\"!?-()\"))
    return cjk or (latin / max(1, len(body)) < 0.7)
bad = sum(1 for r in results if garbage(r[2]))
print(f'  garbled: {bad}/8  => {\"T5 PASS\" if bad == 0 else \"T5 FAIL\"}')
"
