"""Step2: 既存50ep + 追加50ep をマージして 100ep の学習データセットを作る。

src/merge_dataset_v30.py の main() をそのまま利用する。
出力: datasets/soundDiff-m4-f10-s2-p0-nx{m}_ep100
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# merge_dataset_v30 は "datasets" を相対パスで参照するため、プロジェクト直下で実行する
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from merge_dataset_v30 import MergeConfig, main as merge_main  # noqa: E402

BASE_TASK = "soundDiff-m4-f10-s2-p0"
MERGED_SUFFIX = "ep100"
MODES = (0, 1, 2)
TARGET_EPISODES = 100


def episode_count(dataset_dir: Path) -> int:
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    return int(info["total_episodes"])


SOURCE_INDICES = (0, 1)  # _0 = 既存50ep, _1 = 今回追加収集した50ep


def source_datasets(mode: int) -> list[str]:
    """マージ対象の素材データセット名を返す（採番を明示し、想定外の混入を防ぐ）。"""
    names = [f"{BASE_TASK}-nx{mode}_{i}" for i in SOURCE_INDICES]
    missing = [n for n in names if not (PROJECT_ROOT / "datasets" / n).is_dir()]
    if missing:
        print(f"[merge] ERROR: 素材データセットがありません: {missing}")
        return []
    return names


def main() -> int:
    for mode in MODES:
        merged_name = f"{BASE_TASK}-nx{mode}_{MERGED_SUFFIX}"
        merged_dir = PROJECT_ROOT / "datasets" / merged_name

        if merged_dir.exists():
            print(f"[merge] skip (既存): {merged_name} ({episode_count(merged_dir)} ep)")
            continue

        names = source_datasets(mode)
        if not names:
            print(f"[merge] ERROR: 素材データセットが見つかりません (mode={mode})")
            return 1

        total = sum(episode_count(PROJECT_ROOT / "datasets" / n) for n in names)
        print(f"[merge] mode={mode}: {names} -> {merged_name} (合計 {total} ep)")
        if total != TARGET_EPISODES:
            print(f"[merge] WARNING: 合計が {TARGET_EPISODES} ep ではありません ({total} ep)")

        merge_main(MergeConfig(name_list=names, merged_name=merged_name))
        print(f"[merge] done: {merged_name} ({episode_count(merged_dir)} ep)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
