"""verify_spotforming.py の結果を集計・可視化するスクリプト"""

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODE_ORDER = ["spotforming", "beamform_only", "single_mic", "mic_average"]
MODE_LABEL = {
    "spotforming": "Spotforming (ours)",
    "beamform_only": "Steered beamforming (no NMF)",
    "single_mic": "Single mic",
    "mic_average": "Mic average",
}
MODE_COLOR = {"spotforming": "#d62728", "beamform_only": "#ff7f0e", "single_mic": "#1f77b4", "mic_average": "#2ca02c"}
COND_LABEL = {
    "opposite": "Opposite task sound (directional)",
    "white_point": "White noise point source (directional)",
    "sensor": "Sensor white noise (incoherent)",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="workspace/spotforming_verify")
    args = parser.parse_args()

    data = json.load(open(os.path.join(args.in_dir, "records.json")))
    records = data["records"]
    alphas = sorted({r["alpha"] for r in records if r["condition"] != "clean"})
    conditions = [c for c in ["opposite", "white_point", "sensor"]
                  if any(r["condition"] == c for r in records)]

    # 較正: score = corr_A - corr_B、クリーン条件のクラス平均の中点を閾値にする
    # （エンコーダは学習分布で決定境界を較正できるため、未較正argmaxより妥当な下界）
    thresholds = {}
    for m in MODE_ORDER:
        clean_scores = {"A": [], "B": []}
        for r in records:
            if r["condition"] == "clean" and r["mode_name"] == m and r["pred"] is not None:
                clean_scores[r["true_sound"]].append(r["corr_A"] - r["corr_B"])
        thresholds[m] = (np.mean(clean_scores["A"]) + np.mean(clean_scores["B"])) / 2.0

    # (condition, alpha, mode) -> [correct...], [margin...]
    acc = defaultdict(list)
    margin = defaultdict(list)
    for r in records:
        key = (r["condition"], r["alpha"], r["mode_name"])
        if r["pred"] is None:
            acc[key].append(0.0)
            continue
        score = r["corr_A"] - r["corr_B"]
        pred = "A" if score > thresholds[r["mode_name"]] else "B"
        acc[key].append(1.0 if pred == r["true_sound"] else 0.0)
        m = score - thresholds[r["mode_name"]]
        if r["true_sound"] == "B":
            m = -m
        margin[key].append(m)

    # ---- テキストサマリ ----
    lines = []
    n_clean = len(acc[("clean", 0.0, "spotforming")])
    lines.append(f"N per cell = {n_clean}")
    lines.append("")
    header = f"{'condition':<12} {'alpha':>6} | " + " | ".join(f"{m:>12}" for m in MODE_ORDER)
    lines.append(header)
    lines.append("-" * len(header))
    for cond in ["clean"] + conditions:
        cond_alphas = [0.0] if cond == "clean" else alphas
        for a in cond_alphas:
            row = f"{cond:<12} {a:>6g} | "
            cells = []
            for m in MODE_ORDER:
                vals = acc[(cond, a, m)]
                cells.append(f"{np.mean(vals) * 100:>11.1f}%" if vals else f"{'--':>12}")
            lines.append(row + " | ".join(cells))
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(args.in_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")

    # ---- 精度プロット ----
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 4), sharey=True)
    if len(conditions) == 1:
        axes = [axes]
    for ax, cond in zip(axes, conditions):
        for m in MODE_ORDER:
            xs, ys = [], []
            clean_vals = acc[("clean", 0.0, m)]
            if clean_vals:
                xs.append(0.125)  # log軸用にクリーンを左端に置く
                ys.append(np.mean(clean_vals) * 100)
            for a in alphas:
                vals = acc[(cond, a, m)]
                if vals:
                    xs.append(a)
                    ys.append(np.mean(vals) * 100)
            ax.plot(xs, ys, "o-", label=MODE_LABEL[m], color=MODE_COLOR[m])
        ax.axhline(50, color="gray", ls="--", lw=1, label="Chance")
        ax.set_xscale("log")
        ax.set_xticks([0.125] + alphas)
        ax.set_xticklabels(["clean"] + [f"{a:g}" for a in alphas])
        ax.set_xlabel("Interference-to-signal ratio (received RMS)")
        ax.set_title(COND_LABEL.get(cond, cond), fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Sound A/B classification accuracy [%]")
    axes[0].set_ylim(0, 105)
    axes[-1].legend(fontsize=8, loc="lower left")
    fig.suptitle("Task-sound discriminability from spectrogram observation", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.in_dir, "accuracy.png"), dpi=150)

    # ---- マージンプロット ----
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 4), sharey=True)
    if len(conditions) == 1:
        axes = [axes]
    for ax, cond in zip(axes, conditions):
        for m in MODE_ORDER:
            xs, ys, es = [], [], []
            cv = margin[("clean", 0.0, m)]
            if cv:
                xs.append(0.125)
                ys.append(np.mean(cv))
                es.append(np.std(cv) / np.sqrt(len(cv)))
            for a in alphas:
                vals = margin[(cond, a, m)]
                if vals:
                    xs.append(a)
                    ys.append(np.mean(vals))
                    es.append(np.std(vals) / np.sqrt(len(vals)))
            ax.errorbar(xs, ys, yerr=es, fmt="o-", label=MODE_LABEL[m], color=MODE_COLOR[m], capsize=3)
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xticks([0.125] + alphas)
        ax.set_xticklabels(["clean"] + [f"{a:g}" for a in alphas])
        ax.set_xlabel("Interference-to-signal ratio (received RMS)")
        ax.set_title(COND_LABEL.get(cond, cond), fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Template-matching margin (correct - wrong corr.)")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Discrimination margin (mean ± SE)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.in_dir, "margin.png"), dpi=150)

    # ---- サンプルスペクトログラム画像グリッド ----
    sample_dir = os.path.join(args.in_dir, "samples")
    for cond in conditions:
        show_alphas = [a for a in alphas if a in (1.0, 4.0, 8.0)] or alphas[-2:]
        cols = ["clean"] + [f"{cond} a={a:g}" for a in show_alphas]
        fig, axes = plt.subplots(
            len(MODE_ORDER), len(cols), figsize=(2.2 * len(cols), 2.2 * len(MODE_ORDER))
        )
        for i, m in enumerate(MODE_ORDER):
            for j, col in enumerate(cols):
                ax = axes[i][j]
                if col == "clean":
                    pattern = f"clean_a0_{m}_A_t0.npy"
                else:
                    a = show_alphas[j - 1]
                    pattern = f"{cond}_a{a:g}_{m}_A_t0.npy"
                files = glob.glob(os.path.join(sample_dir, pattern))
                if files:
                    ax.imshow(np.load(files[0]), cmap="magma", origin="lower", aspect="auto")
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(col, fontsize=9)
                if j == 0:
                    ax.set_ylabel(MODE_LABEL[m], fontsize=8)
        fig.suptitle(f"Spectrogram observations (sound A): {COND_LABEL.get(cond, cond)}", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.in_dir, f"samples_{cond}.png"), dpi=150)

    print(f"\nplots saved to {args.in_dir}/")


if __name__ == "__main__":
    main()
