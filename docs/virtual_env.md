# Docker環境

### 前提

- NVIDIA GPU が利用でき、`docker` と NVIDIA Container Toolkit が入っていること
- `datasets/` と `outputs/` はホスト側のディレクトリをそのまま使うこと

`./docker/run.sh` は、指定したイメージがまだ無ければ自動で `Dockerfile` からビルドします。

### 初回ログイン

```bash
./docker/run.sh login
```

以下は存在する場合に自動でマウントされるため、認証情報を再利用できます。

- `${HOME}/.cache/huggingface`
- `${HOME}/.cache/wandb`
- `${HOME}/.config/wandb`
- `${HOME}/.netrc`

### 学習と評価

学習してからそのまま評価する例です。

```bash
./docker/run.sh train-eval \
  --dataset-name sound-m4-f10-s2-p0_0 \
  --gpu 0 \
  --policy diffusion \
  --seeds 0,1,2 \
  --steps 100000 \
  --save-freq 10000
```

主なオプション:

- `--policy`: `act` / `diffusion` / `vqbet` / `pi0`
- `--save-freq`: チェックポイント保存間隔
- `--batch-size`: ポリシーごとの既定値を上書き
- `--policy-device`: 既定は `cuda`
- `--episode-num`: 評価エピソード数
- `--checkpoint-step`: 評価時に使うチェックポイント。省略時は `steps`
- `--show-viewer`: 評価時に Genesis viewer を表示
- `--extra-train-arg`: `lerobot-train` に追加の引数を渡す
- `--extra-eval-arg`: `src/eval_policy.py` に追加の引数を渡す

複数 seed を指定すると、出力先は `outputs/train/<policy>_<dataset>_seed<seed>` に分かれます。

### 学習のみ

```bash
./docker/run.sh train \
  --dataset-name sound-m4-f10-s1-p0_0 \
  --policy act
```

### 評価のみ

```bash
./docker/run.sh eval \
  --training-name act_sound-m4-f10-s1-p0_0 \
  --dataset-name sound-m4-f10-s1-p0_0 \
  --checkpoint-step 100000 \
  --episode-num 100
```

### ヘルプ

```bash
./docker/run.sh --help
```
