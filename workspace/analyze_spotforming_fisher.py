"""保存された全スペクトログラムに対してFisher重み付き線形読み出しでA/B識別を評価する。

グローバル相関(analyze_spotforming.py)は判別に効く局所特徴（倍音列など）を
薄めてしまうため、参照試行から求めたFisher重み w = (μA-μB)/(σ_within+ε) による
線形読み出しで評価する。これは学習されたエンコーダが獲得しうる線形判別器の近似。
閾値はクリーンなテスト試行のスコア分布の中点で較正する（学習分布での較正に相当）。
"""

import argparse
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
MODE_COLOR = {
    "spotforming": "#d62728",
    "beamform_only": "#ff7f0e",
    "single_mic": "#1f77b4",
    "mic_average": "#2ca02c",
}
COND_LABEL = {
    "opposite": "Opposite task sound (directional)",
    "white_point": "White noise point source (directional)",
    "sensor": "Sensor white noise (incoherent)",
}


def zscore_img(x):
    x = x.astype(np.float64)
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="workspace/spotforming_verify_4mode")
    args = parser.parse_args()

    data = json.load(open(os.path.join(args.in_dir, "records.json")))
    records = data["records"]
    ref_meta = data["ref_meta"]
    specs = np.load(os.path.join(args.in_dir, "specs.npy")).astype(np.float32)

    modes = [m for m in MODE_ORDER if any(r["mode_name"] == m for r in records)]
    alphas = sorted({r["alpha"] for r in records if r["condition"] != "clean"})
    conditions = [c for c in ["opposite", "white_point", "sensor"]
                  if any(r["condition"] == c for r in records)]

    # ---- Fisher重みの構築（参照試行のみ使用） ----
    weights = {}
    for m in modes:
        cls = {"A": [], "B": []}
        for rm in ref_meta:
            if rm["mode_name"] == m and rm["spec_idx"] >= 0:
                cls[rm["sound"]].append(zscore_img(specs[rm["spec_idx"]]))
        a = np.stack(cls["A"])
        b = np.stack(cls["B"])
        mu_a, mu_b = a.mean(0), b.mean(0)
        sigma = np.sqrt((a.var(0) + b.var(0)) / 2.0)
        eps = 0.5 * np.median(sigma)
        weights[m] = (mu_a - mu_b) / (sigma + eps)

    def score(rec):
        if rec["spec_idx"] < 0:
            return None
        return float(np.mean(zscore_img(specs[rec["spec_idx"]]) * weights[rec["mode_name"]]))

    # ---- 閾値較正（クリーンなテスト試行の中点） ----
    thresholds = {}
    for m in modes:
        cs = {"A": [], "B": []}
        for r in records:
            if r["condition"] == "clean" and r["mode_name"] == m:
                s = score(r)
                if s is not None:
                    cs[r["true_sound"]].append(s)
        thresholds[m] = (np.mean(cs["A"]) + np.mean(cs["B"])) / 2.0

    acc = defaultdict(list)
    margin = defaultdict(list)
    for r in records:
        key = (r["condition"], r["alpha"], r["mode_name"])
        s = score(r)
        if s is None:
            acc[key].append(0.0)
            continue
        pred = "A" if s > thresholds[r["mode_name"]] else "B"
        acc[key].append(1.0 if pred == r["true_sound"] else 0.0)
        mg = s - thresholds[r["mode_name"]]
        if r["true_sound"] == "B":
            mg = -mg
        margin[key].append(mg)

    # クラス内の正規化: マージンをクリーンなマージン平均で割って相対化
    clean_margin = {m: np.mean(margin[("clean", 0.0, m)]) for m in modes}

    lines = [f"N per cell = {len(acc[('clean', 0.0, modes[0])])}", ""]
    header = f"{'condition':<12} {'alpha':>6} | " + " | ".join(f"{m:>13}" for m in modes)
    lines.append(header)
    lines.append("-" * len(header))
    for cond in ["clean"] + conditions:
        cond_alphas = [0.0] if cond == "clean" else alphas
        for a in cond_alphas:
            cells = []
            for m in modes:
                vals = acc[(cond, a, m)]
                cells.append(f"{np.mean(vals) * 100:>12.1f}%" if vals else f"{'--':>13}")
            lines.append(f"{cond:<12} {a:>6g} | " + " | ".join(cells))
    lines.append("")
    lines.append("relative margin (1.0 = clean):")
    for cond in conditions:
        for a in alphas:
            cells = []
            for m in modes:
                vals = margin[(cond, a, m)]
                cells.append(
                    f"{np.mean(vals) / clean_margin[m]:>12.2f} " if vals else f"{'--':>13}"
                )
            lines.append(f"{cond:<12} {a:>6g} | " + " | ".join(cells))
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(args.in_dir, "summary_fisher.txt"), "w") as f:
        f.write(summary + "\n")

    # ---- プロット ----
    for name, table, ylabel, ylim in [
        ("accuracy_fisher", acc, "Sound A/B classification accuracy [%]", (0, 105)),
        ("margin_fisher", margin, "Discrimination margin (relative to clean)", None),
    ]:
        fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 4), sharey=True)
        if len(conditions) == 1:
            axes = [axes]
        for ax, cond in zip(axes, conditions):
            for m in modes:
                xs, ys, es = [], [], []
                for a, vals in [(0.125, table[("clean", 0.0, m)])] + [
                    (a, table[(cond, a, m)]) for a in alphas
                ]:
                    if not vals:
                        continue
                    v = np.asarray(vals, dtype=np.float64)
                    if name.startswith("accuracy"):
                        xs.append(a); ys.append(v.mean() * 100); es.append(0)
                    else:
                        xs.append(a)
                        ys.append(v.mean() / clean_margin[m])
                        es.append(v.std() / np.sqrt(len(v)) / clean_margin[m])
                if name.startswith("accuracy"):
                    ax.plot(xs, ys, "o-", label=MODE_LABEL[m], color=MODE_COLOR[m])
                else:
                    ax.errorbar(xs, ys, yerr=es, fmt="o-", label=MODE_LABEL[m],
                                color=MODE_COLOR[m], capsize=3)
            ax.axhline(50 if name.startswith("accuracy") else 0, color="gray", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_xticks([0.125] + alphas)
            ax.set_xticklabels(["clean"] + [f"{a:g}" for a in alphas])
            ax.set_xlabel("Interference-to-signal ratio (received RMS)")
            ax.set_title(COND_LABEL.get(cond, cond), fontsize=10)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(ylabel)
        if ylim:
            axes[0].set_ylim(*ylim)
        axes[-1].legend(fontsize=8, loc="lower left")
        fig.suptitle("Fisher-weighted linear readout", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(args.in_dir, f"{name}.png"), dpi=150)

    print(f"\nplots saved to {args.in_dir}/")


if __name__ == "__main__":
    main()
