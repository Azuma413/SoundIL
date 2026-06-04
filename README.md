# SoundIL: 音環境情報に基づくピックアンドプレースの模倣学習
## 概要
このプロジェクトは，LeRobotライブラリを基盤として，ACTやDiffusion Policyといった模倣学習モデルに対して，音環境に基づく行動を学習させる手法の実装を目指しています．音環境の認識にはマイクロフォンアレイを利用します．また，音情報のシミュレーション及び処理にはPyroomacousticsライブラリを利用しています．

## フォルダ構成
```
.
├─ docker/                Docker用のスクリプトなど
├─ env/
│  ├─ tasks/              個別タスクを定義
│  └─ genesis_env.py      シミュレーション環境の定義
├─ images/                シミュレータ用のテクスチャを格納
├─ libs/                  ライブラリを格納
│  ├─ Genesis/            Genesis
│  └─ lerobot             LeRobot
├─ sounds/                シミュレータ用の音源を格納
├─ src/
│  ├─ eval_policy.py      評価用スクリプト
│  ├─ iloha_eval.py       実機評価用スクリプト
│  ├─ iloha_server.py     実機データ収集用スクリプト
│  ├─ make_sim_dataset.py データ作成用スクリプト
│  ├─ plot_tsne.py        t-SNEプロット用スクリプト
│  └─ soundreal_utils.py  実機音処理用スクリプト
├─ URDF/                  シミュレーション用URDF
└─ その他設定・管理ファイル
```

## Setup
### システム要件
- Ubuntu 24.04
- CUDA対応のNvidia GPU
### 環境のセットアップ
- 必要なパッケージのインストール
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install portaudio19-dev ffmpeg -y
```
- レポジトリのクローンと環境構築
```bash
git clone -b dev/corl --recurse-submodules https://github.com/Azuma413/SoundIL.git && cd SoundIL
uv sync
```

## シミュレーション実験
### データ収集
```bash
uv run src/make_sim_dataset.py
```

### 学習の実行
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

### 評価
```bash
uv run src/eval_policy.py
```

### t-SNEプロット
`color-by`は [`sound_type`, `sound_coordinate`, `success`] の中から選べる．選ばなければ全部プロットする．

`hidden-reduction`は`none`ならhiddenをstepごとに分割し，それぞれを点として使い，`first`は先頭，`last`は末尾，`mean`はhiddenのstep方向平均を1点にする．
```bash
uv run src/plot_tsne.py \
  --training-name act_sound-m4-f10-s2-p0_0_seed0 \
  --checkpoint-step 100000 \
  --episode-num 100 \
  --hidden-reduction mean
```

## 実機実験
### データ収集
[こちら](https://github.com/KyotoVLATech/AlohaController)で公開されているUnity AppをMeta Quest 3にインストールする。
```bash
uv run src/iloha_server.py
```
上記のコマンドを実行中に、App内でConnectボタンを押すと、QuestとPCを接続できる。
Questのコントローラを用いてロボットを制御できる。

### 評価
- L&I task
```bash
uv run src/iloha_eval.py --policy_path outputs/train/act_soundRealAll-m4-f10-s2-p0_seed0/checkpoints/200000/pretrained_model --dataset_path datasets/soundRealAll-m4-f10-s2-p0 --output_root datasets/eval --episode_time_s 15 --num_episodes 25 --save_data --sound_index 0 --speaker right
```

- Exploratory task
```bash
uv run src/iloha_eval.py --policy_path outputs/train/act_soundRealShake-m4-f10-s2-p0_seed0/checkpoints/200000/pretrained_model --dataset_path datasets/soundRealShake-m4-f10-s2-p0 --output_root datasets/eval --episode_time_s 20 --num_episodes 25 --save_data --is-sound-shake
```
