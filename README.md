# SoundIL: 音環境情報に基づくピックアンドプレースの模倣学習
- [現実環境用ドキュメント](docs/real.md)
- [シミュレーション環境用ドキュメント](docs/sim.md)
## 概要
このプロジェクトは，LeRobotライブラリを基盤として，ACTやDiffusion Policyといった模倣学習モデルに対して，音環境に基づく行動を学習させる手法の実装を目指しています．音環境の認識にはマイクロフォンアレイを利用します．また，音情報のシミュレーション及び処理にはPyroomacousticsライブラリを利用しています．

## フォルダ構成
```
.
├─ datasets/              LeRobot形式のデータセット
├─ docs/                  ドキュメント類
├─ Genesis/               Genesis
├─ lerobot/               LeRobot
├─ outputs/               学習結果，評価結果を格納
├─ URDF/                  シミュレーション用URDF
├─ env/
│  ├─ tasks/              個別タスクを定義
│  └─ genesis_env.py      シミュレーション環境の定義
├─ src/
│  ├─ eval_policy.py      評価用スクリプト
│  └─ make_sim_dataset.py データ作成用スクリプト
└─ その他設定・管理ファイル
```

## Setup
### システム要件
- Ubuntu 24.04
- CUDA対応のNvidia GPU
### 環境のセットアップ
```bash
git clone -b dev/sound --recurse-submodules https://github.com/Azuma413/SoundIL.git
cd SoundIL
uv sync
uv pip install -e "Genesis"
uv pip install -e "lerobot/[smolvla, pi]"
# Linuxなら多分下はやらなくて良い
uv pip uninstall torch torchvision
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
- ffmpegのインストール
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install ffmpeg -y
```

## 学習の実行
先にwandbにログインしてください．
```bash
wandb login
```
POLICYはact, diffusion, vqbet, pi0のいずれかを指定します．
```bash
export DATASET_NAME=[データセット名]
export POLICY=act
uv run lerobot-train \
  --dataset.repo_id=local/${DATASET_NAME} \
  --dataset.root=datasets/${DATASET_NAME} \
  --policy.type=$POLICY \
  --output_dir=outputs/train/${POLICY}_${DATASET_NAME} \
  --job_name=${POLICY}_${DATASET_NAME} \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --batch_size=8 \
  --steps=100000
```

diffusionはbatch size64で100000 step学習