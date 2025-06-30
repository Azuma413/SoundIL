# SoundIL
音環境認識のための模倣学習の実装．\
[現実環境用ドキュメント](docs/real.md)，[シミュレーション環境用ドキュメント](docs/sim.md)

## フォルダ構成
- `datasets/`         LeRobot形式のデータセット
- `docs/`             ドキュメント類
- `Genesis/`          Genesis
- `lerobot/`          LeRobot
- `outputs/`          学習結果，評価結果を格納
- `src/`              評価・データセット生成等のスクリプト
- `URDF/`             シミュレーション用URDF
- その他設定・管理ファイル（`.env`, `.gitignore`, `pyproject.toml`等）

## Setup
- 環境のセットアップ
```bash
git clone --recurse-submodules https://github.com/Azuma413/SoundIL.git
cd SoundIL
uv sync
uv pip install -e "Genesis/[dev]"
uv pip install -e "lerobot/[feetech]"
uv pip uninstall torch torchvision
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
- ffmpegのインストール
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install ffmpeg -y
```

## 学習の実行
先にwandbにログインしておく
```bash
wandb login
```
policyはact, diffusion, pi0, pi0fast, tdmpc, vqbetのいずれか。
学習の安定性を高めるためにbatch sizeはVRAMが許す限り大きくした方が良い。
```bash
export DATASET_NAME=[データセット名]
export POLICY=act
uv run lerobot/lerobot/scripts/train.py \
  --dataset.repo_id=local/${DATASET_NAME} \
  --dataset.root=datasets/${DATASET_NAME} \
  --policy.type=$POLICY \
  --output_dir=outputs/train/${POLICY}-${DATASET_NAME} \
  --job_name=${POLICY}-${DATASET_NAME} \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --batch_size=8 \
  --steps=100000
```
オプション
```bash
  --env.type=test \
  --env.task=test \
  --eval_freq=5000 \
  --eval.n_episodes=10 \
  --eval.batch_size=1
```
- stepsとepochの関係
例えば30fpsで60秒のデータセットを50個用意した場合、全体で90000フレームになるので1epoch=90000sampleとなる。
ここでbatch_sizeを8としていた場合、1stepの学習で8sampleが消費されるため、1epoch=1125stepsとなる。
```
steps = エポック数 * (データfps * データ長さ * データ数) / バッチサイズ
```

学習を再開するときは以下のようにする。
```bash
uv run lerobot/lerobot/scripts/train.py \
  --config_path=outputs/train/act_so100_test/checkpoints/last/pretrained_model/train_config.json \
  --resume=true \
  --steps=150000
```

## Memo
Genesisでcudaを使うにはcuda toolkitが必要。
- ~/.bashrc
```bash
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
```
