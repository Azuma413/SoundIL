"""Step5: 評価結果 (success_rate.txt) を集計してCSVにまとめる。

出力:
  noise_exp/results/raw.csv      条件 x シードごとの成功率
  noise_exp/results/summary.csv  シード3本の平均・標準偏差
"""

import csv
import re
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_ROOT / "noise_exp" / "results"

BASE_TASK = "soundDiff-m4-f10-s2-p0"
MERGED_SUFFIX = "ep100"
POLICY = "act"
STEPS = "100000"
MODES = (0, 1, 2)
SEEDS = (0, 1, 2)
NI_LIST = ("0.0", "0.25", "0.5", "0.75", "1.0", "1.25", "1.5")

MODE_LABEL = {0: "spotforming", 1: "single_mic", 2: "average"}

RATE_RE = re.compile(r"Success rate:\s*(\d+)\s*/\s*(\d+)")


def read_success(path: Path):
    match = RATE_RE.search(path.read_text())
    if not match:
        return None
    success, total = int(match.group(1)), int(match.group(2))
    return success, total


def conditions(mode: int):
    """(env_task, noise_label, ni) を列挙する。"""
    for ni in NI_LIST:
        if ni == "0.0":
            # ノイズ強度0では音源が付加されないためホワイト/反対音は同一。
            # 両系列の基準点として共通の結果を割り当てる。
            task = f"{BASE_TASK}-no{mode}-ni{ni}"
            yield task, "white", ni
            yield task, "opposite", ni
        else:
            yield f"{BASE_TASK}-no{mode}-ni{ni}", "white", ni
            yield f"{BASE_TASK}-no{mode}-ni{ni}-nopp", "opposite", ni


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []

    for mode in MODES:
        for env_task, noise_label, ni in conditions(mode):
            for seed in SEEDS:
                training_name = f"{POLICY}_{BASE_TASK}-nx{mode}_{MERGED_SUFFIX}_seed{seed}"
                path = (
                    PROJECT_ROOT
                    / "outputs"
                    / "eval"
                    / f"{training_name}_{STEPS}_{env_task}"
                    / "success_rate.txt"
                )
                if not path.exists():
                    missing.append(f"{training_name} / {env_task}")
                    continue
                parsed = read_success(path)
                if parsed is None:
                    missing.append(f"{training_name} / {env_task} (パース失敗)")
                    continue
                success, total = parsed
                rows.append(
                    {
                        "condition": MODE_LABEL[mode],
                        "mode": mode,
                        "noise_type": noise_label,
                        "rms_ratio": float(ni),
                        "seed": seed,
                        "success": success,
                        "episodes": total,
                        "success_rate": success / total if total else 0.0,
                        "env_task": env_task,
                    }
                )

    raw_path = RESULT_DIR / "raw.csv"
    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition", "mode", "noise_type", "rms_ratio",
                "seed", "success", "episodes", "success_rate", "env_task",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        grouped.setdefault(
            (row["condition"], row["mode"], row["noise_type"], row["rms_ratio"]), []
        ).append(row["success_rate"])

    summary_path = RESULT_DIR / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["condition", "mode", "noise_type", "rms_ratio", "n_seeds", "mean", "std"]
        )
        for key in sorted(grouped, key=lambda k: (k[1], k[2], k[3])):
            values = grouped[key]
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            writer.writerow([*key, len(values), f"{statistics.mean(values):.4f}", f"{std:.4f}"])

    print(f"[summarize] raw    -> {raw_path}  ({len(rows)} 行)")
    print(f"[summarize] summary-> {summary_path}  ({len(grouped)} 条件)")
    if missing:
        print(f"[summarize] 未完了 {len(missing)} 件:")
        for item in missing[:20]:
            print(f"  - {item}")
        if len(missing) > 20:
            print(f"  ... 他 {len(missing) - 20} 件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
