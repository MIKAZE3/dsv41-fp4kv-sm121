#!/bin/bash
# T6 重测（无 CUDA_LAUNCH_BLOCKING）
docker exec dsv41-head python3 -c "
import json, time, urllib.request, statistics
def loads():
    return json.load(urllib.request.urlopen('http://127.0.0.1:8888/v1/loads', timeout=20))['loads'][0]
def snap():
    d = loads()
    return d['decode_moments'][0], d['decode_moments'][2]
steps = []
for i in range(8):
    for _ in range(30):
        if loads()['num_running_reqs'] == 0: break
        time.sleep(1)
    req = urllib.request.Request('http://127.0.0.1:8888/v1/chat/completions',
        data=json.dumps({'model':'deepseek-v4.1-flash','temperature':0,'max_tokens':256,
        'chat_template_kwargs':{'thinking':False},
        'messages':[{'role':'user','content':f'Story {i}: write about a lighthouse keeper.'}]}).encode(),
        headers={'Content-Type':'application/json'})
    a = snap()
    with urllib.request.urlopen(req, timeout=600) as r:
        json.loads(r.read().decode())
    b = snap()
    dn = b[0]-a[0]; dus = b[1]-a[1]
    if dn > 0:
        steps.append(dus/dn/1000)
        print(f'  round {i+1}: step={dus/dn/1000:.2f} ms')
med = statistics.median(steps)
print(f'  ★ median step = {med:.2f} ms （T6 门 <33.64 ms；V4 基线 32.98；带 blocking 时 37.64）')
print(f'  => {\"T6 PASS\" if med < 33.641 else \"T6 FAIL (+\" + str(round((med/32.98-1)*100,1)) + \"%)\"}')
"
