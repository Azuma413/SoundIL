# sound_dp
音環境認識DiffusionPolicy

Python3.10

## Setup
- 環境のセットアップ
```bash
git clone --recurse-submodules https://github.com/Azuma413/sound_dp.git
cd sound_dp
uv sync
uv pip install torch torchvision torchaudio
cd Genesis
uv pip install -e ".[dev]"
cd ../lerobot
uv pip install -e ".[feetech]"
```
- USBデバイスのセットアップ
Follower用とLeader用のサーボドライバをそれぞれPCに接続し、以下のコマンドを実行する。
```bash
# サーボドライバが/dev/ttyACM0として認識されている場合
udevadm info --name=/dev/ttyACM0 --attribute-walk
```
表示されるリストから、`idProduct`、`idVendor`、`serial`の項目を見つけて値をメモする。
```bash
sudo nano /etc/udev/rules.d/99-usb-devices.rules
```
`idProduct`、`idVendor`、`serial`の項目をメモした値に置き換えて、以下の内容を書き込む。
```
# Follower
SUBSYSTEM=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="xxxxxx", SYMLINK+="follower-driver"
# Leader
SUBSYSTEM=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="xxxxxx", SYMLINK+="leader-driver"
```
udevadmを更新し、内容が適用されているか確認する。
```
sudo udevadm control --reload-rules
sudo udevadm trigger
ls -l /dev/*-driver
```
次に`lerobot/lerobot/common/robot_devices/robots/configs.py`の`So100RobotConfig`を編集する。
```python
class So100RobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/so100"
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyACM1", # 変更
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyACM0", # 変更
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

```
これは正常に動作しない
- モーターのセットアップ
ドライバにボーレートとIDを設定したいモーターを1つ接続した状態で以下のコマンドを実行する。
```
uv run lerobot/lerobot/scripts/configure_motor.py --port /dev/ttyACM0 --brand feetech --model sts3215 --baudrate 1000000 --ID 1
```
`Permission denied`と表示される際は以下のコマンドで権限を付与しておく。
```bash
sudo chmod 666 /dev/ttyACM0
```
- キャリブレーション
```bash
python lerobot/lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_follower"]'
```
```bash
python lerobot/lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_leader"]'
```
- 動作確認
```bash
python lerobot/lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=teleoperate
```
- カメラの確認
```bash
sudo apt install v4l2loopback-dkms v4l-utils
v4l2-ctl --list-devices
python lerobot/lerobot/common/robot_devices/cameras/opencv.py \
    --images-dir outputs/images_from_opencv_cameras
```
使いたいカメラに合わせて`lerobot/lerobot/common/robot_devices/robots/configs.py`の`So100RobotConfig`を編集する。
```python
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "webcam": OpenCVCameraConfig(
                camera_index=2,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )
```
以下のコマンドで映像を表示しながら遠隔操作できる。
```bash
python lerobot/lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=teleoperate \
  --control.display_data=true
```
- データセットの作成


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