# シミュレーション環境での動かし方

## データセットの作成
```bash
uv run src/make_sim_dataset.py
```
`libEGL warning: failed to open /dev/dri/renderD128: Permission denied`という表示が出る場合は、以下を実行
```bash
sudo usermod -aG render $USER
```

## ポリシーの評価
評価を回す前に`src/eval_policy.py`のmain関数の引数を変更しておく．
```py
training_name = "act-weighted_sound-ep100_2" # outputs/trainフォルダに配置されている学習フォルダ名
# 入力画像の解像度．Datasetに揃える
observation_height = 480
observation_width = 640
episode_num = 50 # 評価回数
show_viewer = False # GenesisのViewerを表示するかどうか．
checkpoint_step = "100000" # 何step目のcheckpointを使うか
sim_device = "cpu" # cuda or cpu シミュレーションを実行するデバイスを指定
```
`.env`にDiscord Webhook URLを設定することで，評価終了時にDiscordで通知を受け取れる．\
必要ない場合は`NOTIFICATIONS_ENABLED=false`にしておく．
```bash
uv run src/eval_policy.py
```