#!/bin/bash
# 収集 -> マージ -> 学習 -> 評価 -> 集計 を通しで実行する。
# 各ステップは完了済みのものをスキップするので、中断後の再実行で続きから走る。
#
# 例:
#   bash noise_exp/run_all.sh                 # GPU 0,1,2,3 を使用
#   GPUS=2,3 bash noise_exp/run_all.sh        # 空いているGPUだけ使用
#   EVAL_SLOTS_PER_GPU=2 bash noise_exp/run_all.sh
set -eu
cd "$(dirname "$0")/.."
source noise_exp/common.sh

echo "=============================================="
echo " soundDiff ノイズロバスト性実験"
echo "   GPUS=${GPUS}  wandb=${WANDB}"
echo "=============================================="

echo "--- Step1: 追加データ収集 (${COLLECT_EPISODES} ep x 3条件同時) ---"
if [ -d "datasets/${BASE_TASK}-nx0_1" ]; then
  echo "skip (既に追加データがあります): datasets/${BASE_TASK}-nx0_1"
else
  bash noise_exp/01_collect.sh
fi

echo "--- Step2: マージ (50ep + 50ep -> 100ep) ---"
$UV python noise_exp/02_merge.py

echo "--- Step3: 学習 (3条件 x 3シード = 9本) ---"
bash noise_exp/03_train.sh

echo "--- Step4: 評価 (9モデル x 13ノイズ条件 = 117本) ---"
bash noise_exp/04_eval.sh

echo "--- Step5: 集計 ---"
$UV python noise_exp/05_summarize.py

echo "=============================================="
echo " 完了"
echo "=============================================="
