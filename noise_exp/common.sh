#!/bin/bash
# soundDiff ノイズロバスト性実験の共通設定
#
# 実験設計:
#   学習データ : ノイズ無し(-nx) の soundDiff-m4-f10-s2-p0 を 100ep
#                (既存 50ep + 追加収集 50ep をマージ)
#   3条件      : nx0=Spotforming / nx1=単一マイク / nx2=単純平均
#   学習       : ACT, 100000 steps, seed 0/1/2 (データセットは3シードで共用)
#   評価       : ノイズ有り(-no) 環境で RMS比 -ni を振る。
#                ホワイトノイズ(既定) と 反対音(-nopp) の2系列。
#                ノイズ音源は作業空間中心から半径2m円周上のランダム点(既定値)。

export PROJECT_ROOT="/workspace/myproject"

# --- 基本設定 ---------------------------------------------------------------
export BASE_TASK="soundDiff-m4-f10-s2-p0"
export POLICY="act"
export STEPS=100000
export SAVE_FREQ=10000
export BATCH_SIZE=8
export SEEDS=(0 1 2)
export MODES=(0 1 2)                      # 0=Spotforming 1=単一マイク 2=平均

# --- データ -----------------------------------------------------------------
export COLLECT_EPISODES=50                # 追加収集するエピソード数
export MERGED_SUFFIX="ep100"              # マージ後データセット名の末尾
                                          # -> soundDiff-m4-f10-s2-p0-nx0_ep100

# --- 評価 -------------------------------------------------------------------
export EVAL_EPISODES=100
export NI_LIST=(0.0 0.25 0.5 0.75 1.0 1.25 1.5)   # ノイズRMS比

# --- 実行資源 ---------------------------------------------------------------
# 使用するGPU (カンマ区切り)。環境変数 GPUS で上書き可。
# 2026-07-28 時点で GPU0/1 は本コンテナ外のプロセスが占有 (各37GB/util100%) していたため
# 既定は空いている 2,3 のみ。4枚使う場合は GPUS=0,1,2,3 を指定する。
export GPUS="${GPUS:-2,3}"
# ACT(batch_size=8) 1本ではRTX6000 Adaを使い切らないため、1GPUに複数ジョブを載せる。
# VRAM: 学習 約8GB/本、評価 約3GB/本 (48GB中) なので2本同時でも余裕がある。
export TRAIN_SLOTS_PER_GPU="${TRAIN_SLOTS_PER_GPU:-2}"
export EVAL_SLOTS_PER_GPU="${EVAL_SLOTS_PER_GPU:-2}"

# wandb: 未ログインだと学習が落ちるため既定は無効。
# 有効化するには `uv run wandb login` 実行後に WANDB=1 を指定。
export WANDB="${WANDB:-0}"
if [ "$WANDB" = "1" ]; then
  export WANDB_ARGS="--wandb.enable=true --wandb.disable_artifact=true"
else
  export WANDB_ARGS="--wandb.enable=false"
fi

# uv の同期を毎回走らせない (環境は構築済み)
export UV="uv run --no-sync"

export LOG_ROOT="${PROJECT_ROOT}/noise_exp/logs"
mkdir -p "$LOG_ROOT"

# --- ヘルパ -----------------------------------------------------------------
# マージ後のデータセット名
dataset_name() {  # $1 = mode
  echo "${BASE_TASK}-nx$1_${MERGED_SUFFIX}"
}

# 学習ランの名前
training_name() {  # $1 = mode, $2 = seed
  echo "${POLICY}_$(dataset_name "$1")_seed$2"
}

# 評価に使う env-task 文字列を列挙 (mode ごとに13通り)
#   ni=0.0 はノイズ音源自体が生成されない (sound_camera.py: noise_intensity>0 のときのみ付加)
#   ため、ホワイトノイズ版と反対音版は同一。重複を避け1本のみ実行する。
eval_env_tasks() {  # $1 = mode
  local m="$1" ni
  for ni in "${NI_LIST[@]}"; do
    echo "${BASE_TASK}-no${m}-ni${ni}"
    if [ "$ni" != "0.0" ]; then
      echo "${BASE_TASK}-no${m}-ni${ni}-nopp"
    fi
  done
}
