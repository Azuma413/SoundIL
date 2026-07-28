#!/bin/bash
# Stage 2: Spotforming OOD検証パイプライン
#   データセット生成(クリーン, 修正済みSpotforming) → ACT学習(100k) → クリーン評価 → OOD評価
# 使い方: bash workspace/run_spotforming_stage2.sh <spectrogram_mode 0|1|2> <gpu_id>
set -euo pipefail
MODE=$1
GPU=$2
export CUDA_VISIBLE_DEVICES=${GPU}
TASK="soundDiff-m4-f10-s2-p0-nx${MODE}"
DS="${TASK}_0"
cd /workspace/myproject

echo "=== [mode ${MODE}] dataset generation (${TASK}, 50 episodes) ==="
uv run --no-sync python src/make_sim_dataset.py ${TASK} --episode-num 50

echo "=== [mode ${MODE}] ACT training (100k steps) ==="
uv run --no-sync lerobot-train \
  --dataset.repo_id=local/${DS} \
  --dataset.root=datasets/${DS} \
  --policy.type=act \
  --output_dir=outputs/train/act_${DS}_seed0 \
  --job_name=act_${DS}_seed0 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --seed=0 \
  --batch_size=8 \
  --steps=100000

echo "=== [mode ${MODE}] eval: in-distribution (clean) ==="
uv run --no-sync python src/eval_policy.py \
  --training-name act_${DS}_seed0 --checkpoint-step 100000 --episode-num 50

for NI in 0.25 0.5 1.0; do
  echo "=== [mode ${MODE}] eval: OOD opposite-sound noise ni=${NI} ==="
  uv run --no-sync python src/eval_policy.py \
    --training-name act_${DS}_seed0 --checkpoint-step 100000 --episode-num 50 \
    --env-task soundDiff-m4-f10-s2-p0-no${MODE}-ni${NI}-nopp
done

echo "=== [mode ${MODE}] ALL DONE ==="
