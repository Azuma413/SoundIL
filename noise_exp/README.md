# soundDiff ノイズロバスト性実験

ACT / soundDiff タスクで、**音処理3条件**の学習済みポリシーが
**ノイズ音源**にどれだけ頑健かを、シード3本で評価する実験一式。

## 実験設計

| 項目 | 内容 |
|---|---|
| タスク | `soundDiff-m4-f10-s2-p0` |
| ポリシー | ACT (batch_size=8, steps=100000, save_freq=10000) |
| 学習データ | **ノイズ無し** (`-nx`) 100 ep = 既存 50 ep + 追加収集 50 ep をマージ |
| 学習条件 | nx0 = Spotforming / nx1 = 単一マイク / nx2 = 単純平均 |
| 学習シード | 0, 1, 2（データセットは3シードで共用） |
| 評価 | ノイズ有り (`-no`) 環境、各条件 100 エピソード |

### 評価のノイズ条件（学習条件ごとに13通り）

| RMS比 (`-ni`) | ホワイトノイズ | 反対音 (`-nopp`) |
|---|---|---|
| 0.0 | 共通1本 | 共通1本（同上） |
| 0.25 / 0.5 / 0.75 / 1.0 / 1.25 / 1.5 | 各1本 | 各1本 |

`-ni0.0` はノイズ音源そのものが生成されない
（`env/tasks/sound_camera.py` で `noise_intensity > 0` のときのみ音源を付加）ため、
ホワイトノイズ版と反対音版は同一の実験になる。重複実行を避けて1本だけ走らせ、
集計時に両系列の基準点として共有する。

ノイズ音源位置は作業空間中心から**半径2mの円周上のランダム点**
（`noise_source_radius` の既定値2.0、エピソード内で固定）。

合計: 学習 3条件 × 3シード = **9本** / 評価 9モデル × 13条件 = **117本**

## 実行

```bash
# 通し実行（各ステップは完了済みをスキップするので中断後も再開できる）
bash noise_exp/run_all.sh

# 使うGPUを指定
GPUS=0,1,2,3 bash noise_exp/run_all.sh

# 1GPUあたりの同時実行数を上げる
EVAL_SLOTS_PER_GPU=2 bash noise_exp/run_all.sh

# wandbを使う（先に `uv run wandb login` が必要）
WANDB=1 bash noise_exp/run_all.sh
```

個別に回す場合:

```bash
bash noise_exp/01_collect.sh            # 追加50ep収集（3条件同時）
uv run --no-sync python noise_exp/02_merge.py   # 50+50 -> 100ep
bash noise_exp/03_train.sh              # 学習9本
bash noise_exp/04_eval.sh               # 評価117本
uv run --no-sync python noise_exp/05_summarize.py  # 集計
```

## ファイル

| ファイル | 役割 |
|---|---|
| `common.sh` | 共通設定（GPU、シード、RMS比リスト、命名規則） |
| `run_jobs.sh` | GPUスロットへジョブを割り当てる並列キューランナー |
| `01_collect.sh` | `make_sim_dataset.py --all-modes` で3条件を1回で収集 |
| `02_merge.py` | `src/merge_dataset_v30.py` で 50ep + 50ep を100epにマージ |
| `03_train.sh` | 学習9本を生成してGPU並列実行 |
| `04_eval.sh` | 評価117本を生成してGPU並列実行 |
| `05_summarize.py` | `success_rate.txt` を集計して CSV 出力 |

## 出力

- データセット: `datasets/soundDiff-m4-f10-s2-p0-nx{0,1,2}_ep100`
- 学習: `outputs/train/act_soundDiff-m4-f10-s2-p0-nx{m}_ep100_seed{s}`
- 評価: `outputs/eval/<training_name>_100000_<env_task>/success_rate.txt`
- 集計: `noise_exp/results/raw.csv`, `noise_exp/results/summary.csv`
- ログ: `noise_exp/logs/`（`train/`, `eval/` 配下にジョブ別ログ、`_failures.txt` に失敗一覧）

## 注意

- **GPU割り当て**: 既定は `GPUS=2,3`。2026-07-28 時点で GPU0/1 は本コンテナ外の
  プロセスが占有（各37GB・util 100%）していたため。4枚使えるなら `GPUS=0,1,2,3`。
- **wandb**: 既定は無効。未ログイン状態で `--wandb.enable=true` にすると学習が落ちる。
- **環境**: `uv run` は同期時に pyaudio のビルドで失敗しうるため、
  スクリプトは `uv run --no-sync` を使う。環境構築は README.md の手順
  （`portaudio19-dev` + `uv sync` + `uv pip install -e Genesis` + `uv pip install -e lerobot/[pi]`）。
  Genesis の import には `libglu1-mesa` も必要。
