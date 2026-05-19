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
- 必要なパッケージのインストール
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install portaudio19-dev ffmpeg -y
```
- レポジトリのクローンと環境構築
```bash
git clone -b dev/corl --recurse-submodules https://github.com/Azuma413/SoundIL.git
cd SoundIL
uv sync
uv pip install -e "Genesis"
uv pip install -e "lerobot/[pi]"
```
- ログイン
```bash
uv run huggingface-cli login
uv run wandb login
```

## 学習の実行
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
## t-SNEプロット
`color-by`は [`sound_type`, `sound_coordinate`, `success`] の中から選べる．選ばなければ全部プロットする．

`hidden-reduction`は`none`ならhiddenをstepごとに分割し，それぞれを点として使い，`first`は先頭，`last`は末尾，`mean`はhiddenのstep方向平均を1点にする．
```bash
uv run plot_tsne.py \
  --training-name act_sound-m4-f10-s2-p0_0_seed0 \
  --checkpoint-step 100000 \
  --episode-num 100 \
  --hidden-reduction mean
```

## Datasetのupdate
actionを更新する
```bash
uv run update_action.py soundRealShake-m4-f10-s2-p0 -o soundRealShake-shifted
```
actionとstateをvideoに合わせる
```bash
uv run shift_action_state.py 5 soundRealAll-shifted -o soundRealAll-shifted-as
```
edit
```bash
uv run edit_dataset.py
```
actionとstateを元に戻す．
```bash
uv run shift_action_state.py -5 soundRealAll-edited -o soundRealAll-edited-as
```
## 実機評価
- All
```bash
uv run iloha_eval.py --policy_path outputs/train/act_soundRealAll-m4-f10-s2-p0_seed0/checkpoints/200000/pretrained_model --dataset_path datasets/soundRealAll-m4-f10-s2-p0 --output_root datasets/eval --episode_time_s 15 --num_episodes 25 --save_data --sound_index 0 --speaker right
```

- Shake
```bash
uv run iloha_eval.py --policy_path outputs/train/act_soundRealShake-m4-f10-s2-p0_seed0/checkpoints/200000/pretrained_model --dataset_path datasets/soundRealShake-m4-f10-s2-p0 --output_root datasets/eval --episode_time_s 20 --num_episodes 25 --save_data --is-sound-shake
```
TEなし, n_action_steps:30
ブザー奥、ぴよぴよ手前
left奥、right手前

- | r0 | r1 | l0 | l1 | sum
---|---|---|---|---| ---
s2-0|19 |17 |17 |19 |72
s2-1| - | - | - | - |
s2-2| - | - | - | - |
s0-0| 2 | 0 | 3 | 0 |5
s0-1| - | - | - | - |
s0-2| - | - | - | - |

- | 手前 | 奥 | sum
---|---|---|---
s2-0| , |15, |
s2-1| - | - |
s2-2| - | - |
s0-0|4,8|0,0|12
s0-1| - | - |
s0-2| - | - |