# Docker環境

### 前提

- NVIDIA GPU が利用でき、`docker` と NVIDIA Container Toolkit が入っていること
- `datasets/` と `outputs/` はホスト側のディレクトリをそのまま使うこと

`./docker/run.sh` は、指定したイメージがまだ無ければ自動で `Dockerfile` からビルドします。

### 初回ログイン

```bash
./docker/run.sh login
```

以下は自動で作成・マウントされるため、`login` で入力した認証情報は次回以降の `train` / `train-eval` / `eval` でも再利用できます。

- `${HOME}/.cache/huggingface`
- `${HOME}/.cache/wandb`
- `${HOME}/.config/wandb`
- `${HOME}/.netrc`

`pi0` は gated model の `google/paligemma-3b-pt-224` を使うため、Hugging Face 側でモデル利用申請が承認されたアカウントでログインしてください。

### 学習と評価

学習してからそのまま評価する例です。

```bash
./docker/run.sh train-eval \
  --dataset-name sound-m4-f10-s2-p0_0 \
  --gpu 0 \
  --policy act \
  --seeds 0 \
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

seed の数に関係なく、出力先は `outputs/train/<policy>_<dataset>_seed<seed>` になります。

### 学習のみ

```bash
./docker/run.sh train \
  --dataset-name sound-m4-f10-s2-p0_0 \
  --gpu 1 \
  --seeds 2 \
  --steps 100000 \
  --save-freq 10000 \
  --policy vqbet
```

### 評価のみ

```bash
for step in 20000 40000 60000 80000 100000 120000 140000 160000 180000; do
  ./docker/run.sh eval \
    --training-name diffusion_soundShake-m4-f10-s2-p0_0_seed2 \
    --gpu 0 \
    --dataset-name soundShake-m4-f10-s2-p0_0 \
    --checkpoint-step "$step" \
    --episode-num 100
done
```

./docker/run.sh eval \
  --training-name pi0_soundShake-m4-f10-s1-p0_0_seed2 \
  --gpu 0 \
  --dataset-name soundShake-m4-f10-s1-p0_0 \
  --checkpoint-step 200000 \
  --episode-num 100

### ヘルプ

```bash
./docker/run.sh --help
```

### t-SNEプロット
policy, dataset name, gpu indexの順
```bash
./docker/tsne.sh diffusion soundDiff-m4-f10-s2-p0 2
```
