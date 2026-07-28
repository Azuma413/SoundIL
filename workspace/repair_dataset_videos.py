"""データセット内の破損した動画(デコード不能パケットを含むmp4)を検査・修復するスクリプト。

破損パケットをスキップしてデコードし、欠損フレームは直前フレームの複製で埋めて
lerobotと同一設定(libsvtav1, crf30, g2, yuv420p)で再エンコードする。
フレーム数・タイムスタンプは元と完全に一致させる。

使い方: uv run python workspace/repair_dataset_videos.py datasets/<dataset_name> [--fps 30]
"""

import argparse
import glob
import os
import shutil
import subprocess
import tempfile

import av
import numpy as np
from PIL import Image

from lerobot.datasets.video_utils import encode_video_frames


def ffmpeg_check(path: str) -> str:
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "null", "-"],
        capture_output=True, text=True,
    )
    return r.stderr.strip()


def count_packets(path: str) -> int:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return int(r.stdout.strip())


def tolerant_decode(path: str, fps: int):
    """破損パケットをスキップしてフレームを {frame_index: ndarray} で返す"""
    container = av.open(path)
    stream = container.streams.video[0]
    tb = stream.time_base
    frames = {}
    for packet in container.demux(stream):
        try:
            for frame in packet.decode():
                idx = int(round(float(frame.pts * tb) * fps))
                frames[idx] = frame.to_ndarray(format="rgb24")
        except av.error.InvalidDataError:
            continue
    # flush
    try:
        for frame in stream.codec_context.decode(None):
            idx = int(round(float(frame.pts * tb) * fps))
            frames[idx] = frame.to_ndarray(format="rgb24")
    except Exception:
        pass
    container.close()
    return frames


def repair_video(path: str, fps: int) -> bool:
    n_expected = count_packets(path)
    frames = tolerant_decode(path, fps)
    missing = [i for i in range(n_expected) if i not in frames]
    print(f"  expected={n_expected} decoded={len(frames)} missing={len(missing)} {missing[:20]}")
    if not missing and len(frames) == n_expected:
        print("  no missing frames after tolerant decode; re-encoding anyway to clean file")
    with tempfile.TemporaryDirectory(dir=os.path.dirname(path)) as tmp:
        prev = None
        for i in range(n_expected):
            img = frames.get(i)
            if img is None:
                if prev is None:
                    # 先頭欠損: 直後の存在フレームを使う
                    nxt = next((frames[j] for j in range(i + 1, n_expected) if j in frames), None)
                    img = nxt if nxt is not None else np.zeros_like(next(iter(frames.values())))
                else:
                    img = prev
            prev = img
            Image.fromarray(img).save(os.path.join(tmp, f"frame-{i:06d}.png"))
        out_path = path + ".repaired.mp4"
        if os.path.exists(out_path):
            os.remove(out_path)
        encode_video_frames(tmp, out_path, fps, overwrite=True)
    # 検証
    err = ffmpeg_check(out_path)
    n_new = count_packets(out_path)
    if err or n_new != n_expected:
        print(f"  REPAIR FAILED: err={err[:200]} n_new={n_new}")
        os.remove(out_path)
        return False
    shutil.move(path, path + ".corrupt.bak")
    shutil.move(out_path, path)
    print("  repaired OK")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    ok = True
    for path in sorted(glob.glob(os.path.join(args.dataset_root, "videos", "**", "*.mp4"), recursive=True)):
        if path.endswith(".repaired.mp4"):
            continue
        err = ffmpeg_check(path)
        if not err:
            print(f"[ok]      {path}")
            continue
        print(f"[corrupt] {path}")
        if not repair_video(path, args.fps):
            ok = False
    if not ok:
        raise SystemExit(1)
    print("all videos ok")


if __name__ == "__main__":
    main()
