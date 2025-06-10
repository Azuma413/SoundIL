# 2つのデータセットを結合し、indexをずらして保存するスクリプト
# python datasets/concat_data.py path1 path2 save_path
import os
import sys
import shutil
import json
from glob import glob

def get_max_index_from_filenames(file_list, keylen=6):
    # 000123.npz や episode_000123.parquet などから最大indexを取得
    max_idx = -1
    for f in file_list:
        basename = os.path.basename(f)
        nums = [int(s) for s in basename.replace('.', '_').split('_') if s.isdigit() and len(s) >= keylen]
        if nums:
            max_idx = max(max_idx, max(nums))
    return max_idx

def concat_files_with_index_shift(src_dir, dst_dir, start_idx, pattern, keylen=6, prefix='', suffix=''):
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(glob(os.path.join(src_dir, pattern)))
    for f in files:
        basename = os.path.basename(f)
        # index部分を抽出
        nums = [int(s) for s in basename.replace('.', '_').split('_') if s.isdigit() and len(s) >= keylen]
        if not nums:
            continue
        idx = nums[-1]
        new_idx = idx + start_idx
        new_name = basename.replace(f"{idx:0{keylen}d}", f"{new_idx:0{keylen}d}")
        shutil.copy2(f, os.path.join(dst_dir, new_name))

def merge_jsonl_with_index_shift(file1, file2, out_file, key='episode_index', shift=0):
    # file1, file2: 入力jsonlファイル
    # out_file: 出力jsonlファイル
    with open(out_file, 'w') as fout:
        with open(file1) as f1:
            for line in f1:
                fout.write(line)
        with open(file2) as f2:
            for line in f2:
                obj = json.loads(line)
                if key in obj:
                    obj[key] += shift
                fout.write(json.dumps(obj, ensure_ascii=False) + '\n')

def merge_info_json(info1_path, info2_path, out_path, ep_shift, frame_shift, video_shift):
    with open(info1_path) as f1, open(info2_path) as f2:
        info1 = json.load(f1)
        info2 = json.load(f2)
    info1['total_episodes'] += info2['total_episodes']
    info1['total_frames'] += info2['total_frames']
    info1['total_videos'] += info2['total_videos']
    # splits, data_path, video_path, features等はinfo1を優先
    with open(out_path, 'w') as fout:
        json.dump(info1, fout, indent=4, ensure_ascii=False)

def main():
    if len(sys.argv) != 4:
        print('Usage: python concat_data.py path1 path2 save_path')
        sys.exit(1)
    path1, path2, save_path = sys.argv[1:4]
    os.makedirs(save_path, exist_ok=True)
    # data
    data1_dir = os.path.join(path1, 'data/chunk-000')
    data2_dir = os.path.join(path2, 'data/chunk-000')
    save_data_dir = os.path.join(save_path, 'data/chunk-000')
    os.makedirs(save_data_dir, exist_ok=True)
    files1 = sorted(glob(os.path.join(data1_dir, '*'))) # data1_dirのファイルを取得してソート
    files2 = sorted(glob(os.path.join(data2_dir, '*')))
    max_idx1 = get_max_index_from_filenames(files1) # data1_dirの最大indexを取得
    concat_files_with_index_shift(data1_dir, save_data_dir, 0, '*') # data1_dirのファイルをコピー
    concat_files_with_index_shift(data2_dir, save_data_dir, max_idx1+1, '*') # data2_dirのファイルをコピー
    # videos
    # camera_list = ['front', 'side', 'sound']
    camera_list = ['realsense', 'sound', 'webcam']
    for camera in camera_list:
        videos1_dir = os.path.join(path1, f'videos/chunk-000/observation.images.{camera}')
        videos2_dir = os.path.join(path2, f'videos/chunk-000/observation.images.{camera}')
        save_videos_dir = os.path.join(save_path, f'videos/chunk-000/observation.images.{camera}')
        os.makedirs(save_videos_dir, exist_ok=True)
        files1 = sorted(glob(os.path.join(videos1_dir, '*')))
        files2 = sorted(glob(os.path.join(videos2_dir, '*')))
        max_vidx1 = get_max_index_from_filenames(files1)
        concat_files_with_index_shift(videos1_dir, save_videos_dir, 0, '*')
        concat_files_with_index_shift(videos2_dir, save_videos_dir, max_vidx1+1, '*')
    # meta
    meta1_dir = os.path.join(path1, 'meta')
    meta2_dir = os.path.join(path2, 'meta')
    save_meta_dir = os.path.join(save_path, 'meta')
    os.makedirs(save_meta_dir, exist_ok=True)
    # episodes.jsonl
    merge_jsonl_with_index_shift(
        os.path.join(meta1_dir, 'episodes.jsonl'),
        os.path.join(meta2_dir, 'episodes.jsonl'),
        os.path.join(save_meta_dir, 'episodes.jsonl'),
        key='episode_index', shift=max_idx1+1)
    # episodes_stats.jsonl
    merge_jsonl_with_index_shift(
        os.path.join(meta1_dir, 'episodes_stats.jsonl'),
        os.path.join(meta2_dir, 'episodes_stats.jsonl'),
        os.path.join(save_meta_dir, 'episodes_stats.jsonl'),
        key='episode_index', shift=max_idx1+1)
    # tasks.jsonl（重複排除）
    tasks1 = set()
    tasks2 = set()
    with open(os.path.join(meta1_dir, 'tasks.jsonl')) as f1:
        for line in f1:
            tasks1.add(line.strip())
    with open(os.path.join(meta2_dir, 'tasks.jsonl')) as f2:
        for line in f2:
            tasks2.add(line.strip())
    all_tasks = sorted(tasks1 | tasks2)
    with open(os.path.join(save_meta_dir, 'tasks.jsonl'), 'w') as fout:
        for t in all_tasks:
            fout.write(t + '\n')
    # info.json
    merge_info_json(
        os.path.join(meta1_dir, 'info.json'),
        os.path.join(meta2_dir, 'info.json'),
        os.path.join(save_meta_dir, 'info.json'),
        ep_shift=max_idx1+1, frame_shift=0, video_shift=0)

if __name__ == '__main__':
    main()