# sound_dp
音環境認識DiffusionPolicy

Python3.10

## Setup
```bash
git clone --recurse-submodules https://github.com/Azuma413/sound_dp.git
cd sound_dp
uv sync
uv pip install torch torchvision torchaudio
cd Genesis
uv pip install -e ".[dev]"
cd ../lerobot
uv pip install -e .
```

## Run
```bash
uv run -m src.main
```

## [SO-100](lerobot/examples/10_use_so100.md)

## Memo
SO-100のURDFは以下のリポジトリから取ってきた。\
https://github.com/TheRobotStudio/SO-ARM100

## TODO
- [x] ubuntu, wsl上にgenesisの可視化環境を作る
- [x] SO-100のurdfをgenesisに読み込ませる
- [ ] [genesisとpyroomacausticsを組み合わせてgym環境を作成する](https://qiita.com/hbvcg00/items/473d5049dd3fe36d2fa3)
- [ ] [現実のマスターアームを組み立て、LeRobotで値を読み取れるようにする](https://note.com/npaka/n/nf41de358825d)
- [ ] マスターのデータをシミュレーション環境に反映させる
- [ ] データセット作成
- [ ] データセットを利用してDPの学習を行う
- [ ] 現実でデータセット作成環境を構築する
- [ ] 現実でデータセットを作成する
- [ ] データセットを利用してDPの学習を行う
- [ ] 現実で動かしてみる