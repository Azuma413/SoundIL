#!/usr/bin/env python3
import os
import time
import json
import socket
import subprocess
import requests

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1387728890222346250/45o4Xi_Xuozb0fZxADIKgS9pB6gUCt2UDUB8ENkbRo6hLSGQ8wvgPitXmJrl6t5IlTd0"

CHECK_INTERVAL_SEC = 30
REQUIRE_CONSECUTIVE_OK = 2   # 連続2回空きなら通知
HOSTNAME = socket.gethostname()

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()

def get_gpu_info():
    # GPU index と UUID を対応付ける
    out = run([
        "nvidia-smi",
        "--query-gpu=index,uuid,name",
        "--format=csv,noheader,nounits"
    ])
    gpus = {}
    for line in out.splitlines():
        idx, uuid, name = [x.strip() for x in line.split(",", 2)]
        gpus[uuid] = {
            "index": int(idx),
            "uuid": uuid,
            "name": name,
        }
    return gpus

def get_compute_processes():
    # compute プロセスだけ取得する
    # 何もいないと nvidia-smi が空文字を返すことがある
    try:
        out = run([
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits"
        ])
    except subprocess.CalledProcessError:
        return []

    if not out:
        return []

    procs = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",", 3)]
        if len(parts) != 4:
            continue
        gpu_uuid, pid, proc_name, used_mem = parts
        try:
            pid = int(pid)
        except ValueError:
            continue

        used_mem_mb = None
        try:
            used_mem_mb = int(used_mem.split()[0])
        except Exception:
            pass

        procs.append({
            "gpu_uuid": gpu_uuid,
            "pid": pid,
            "process_name": proc_name,
            "used_mem_mb": used_mem_mb,
        })
    return procs

def build_gpu_status():
    gpus = get_gpu_info()
    procs = get_compute_processes()

    # 各GPUに載っている compute プロセス一覧
    by_gpu = {uuid: [] for uuid in gpus.keys()}
    for p in procs:
        if p["gpu_uuid"] in by_gpu:
            by_gpu[p["gpu_uuid"]].append(p)

    status = []
    for uuid, gpu in sorted(gpus.items(), key=lambda kv: kv[1]["index"]):
        status.append({
            "index": gpu["index"],
            "uuid": uuid,
            "name": gpu["name"],
            "compute_processes": by_gpu[uuid],
            "is_free": len(by_gpu[uuid]) == 0,
        })
    return status

def send_discord_message(content):
    resp = requests.post(
        DISCORD_WEBHOOK_URL,
        json={"content": content},
        timeout=10,
    )
    resp.raise_for_status()

def format_free_gpu_message(gpu):
    return (
        f"🟢 GPUが1枚空きました\n"
        f"host: `{HOSTNAME}`\n"
        f"gpu: `{gpu['index']} - {gpu['name']}`\n"
        f"condition: `compute process = 0`"
    )

def main():
    consecutive_free = {}   # gpu_index -> count
    notified = set()        # 通知済みの gpu_index

    while True:
        try:
            status = build_gpu_status()

            for gpu in status:
                idx = gpu["index"]

                if gpu["is_free"]:
                    consecutive_free[idx] = consecutive_free.get(idx, 0) + 1

                    if consecutive_free[idx] >= REQUIRE_CONSECUTIVE_OK and idx not in notified:
                        send_discord_message(format_free_gpu_message(gpu))
                        notified.add(idx)
                else:
                    # 再び使用中になったら、次回空いたとき通知できるようにリセット
                    consecutive_free[idx] = 0
                    if idx in notified:
                        notified.remove(idx)

        except Exception as e:
            try:
                send_discord_message(
                    f"⚠️ GPU監視エラー on `{HOSTNAME}`: `{type(e).__name__}: {e}`"
                )
            except Exception:
                pass

        time.sleep(CHECK_INTERVAL_SEC)

if __name__ == "__main__":
    main()