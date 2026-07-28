# Real-Robot Workflow

This document covers data collection and policy evaluation on the physical robot.
For installation, the simulation workflow, and the naming conventions used below,
see [README.md](README.md).

Everything here assumes you run commands **from the repository root**, because all
paths in the code are relative to it.

> **Note**
> The real-robot scripts are written for one specific lab setup. Serial port names,
> camera serial numbers, and audio device ids are hardcoded in the source. You will
> have to edit them before the scripts run on different hardware — the exact places
> are listed in [Adapting to your hardware](#adapting-to-your-hardware).

## Contents

- [Hardware](#hardware)
- [Adapting to your hardware](#adapting-to-your-hardware)
- [Data collection with the Meta Quest 3](#data-collection-with-the-meta-quest-3)
- [Training on real-robot data](#training-on-real-robot-data)
- [Evaluation on the real robot](#evaluation-on-the-real-robot)
- [Output files](#output-files)
- [Troubleshooting](#troubleshooting)

## Hardware

| Component | What is used | Notes |
| --- | --- | --- |
| Robot | Iloha bimanual arm | Only the **right** arm is enabled (7 DoF + gripper). The left arm is disabled in both scripts. |
| Actuator drivers | Dynamixel + RobStride, USB serial | Four serial ports, see below. |
| Cameras | 2 × Intel RealSense | Selected by serial number, not by index. `front` = `146222252104` (1280×720@30), `side` = `029522250086` (640×480@30). |
| Microphones | 4 × TAMAGO 8-channel arrays | 16 kHz. Default `sounddevice` input ids `[6, 7, 8, 9]`, auto-detected by a device name containing `tamago`. |
| Speakers | Stereo output | The left/right channel selects which physical speaker sounds. |
| Teleoperation | Meta Quest 3 + [KyotoVLATech/AlohaController](https://github.com/KyotoVLATech/AlohaController) | Connects to the PC over Wi-Fi. |
| GPU | CUDA-capable NVIDIA GPU | Used for policy inference and real-time NMF. |

The task name used throughout the real-robot pipeline is fixed to
`soundReal-m4-f10-s2-p0` (`SOUNDREAL_TASK_NAME` in [src/s2a2/soundreal_utils.py:21](src/s2a2/soundreal_utils.py#L21)),
so real-robot datasets are named `soundReal-m4-f10-s2-p0_0`, `soundReal-m4-f10-s2-p0_1`, and so on.

### Audio devices

List the input devices that `sounddevice` can see:

```bash
uv run python -c "import sounddevice; print(sounddevice.query_devices())"
```

If the four TAMAGO arrays are not at ids `6,7,8,9`, pass the ids explicitly
(`--input_device_ids 6,7,8,9` for `iloha_eval.py`) or export them so both scripts
pick them up:

```bash
export SOUNDREAL_DEVICE_IDS=6,7,8,9
```

### Serial ports

The two scripts open these ports:

| Port | `iloha_server.py` | `iloha_eval.py` |
| --- | --- | --- |
| Right Dynamixel | `/dev/ttyUSB_RightDynamixel` | `/dev/ttyUSB_RightDynamixel` |
| Right RobStride | `/dev/ttyUSB0` | `/dev/ttyUSB0` |
| Left RobStride | `/dev/ttyUSB3` | `/dev/ttyUSB1` |
| Left Dynamixel | `/dev/ttyUSB_LeftDynamixel` | `/dev/ttyUSB_LeftDynamixel` |

`/dev/ttyUSB_RightDynamixel` and `/dev/ttyUSB_LeftDynamixel` are **udev symlinks**, not
kernel device names; they do not exist until you create a udev rule. Find the serial
number of each adapter with `udevadm info -a -n /dev/ttyUSB0 | grep serial`, then add
`/etc/udev/rules.d/99-iloha.rules`:

```bash
sudo tee /etc/udev/rules.d/99-iloha.rules > /dev/null <<'EOF'
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{serial}=="REPLACE_WITH_RIGHT_SERIAL", SYMLINK+="ttyUSB_RightDynamixel"
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{serial}=="REPLACE_WITH_LEFT_SERIAL", SYMLINK+="ttyUSB_LeftDynamixel"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Your user also needs permission to open serial ports:

```bash
sudo usermod -aG dialout $USER   # log out and back in for this to take effect
```

## Adapting to your hardware

| What to change | Where |
| --- | --- |
| Serial port names, per-joint limits, gripper current limits | [src/s2a2/iloha_eval.py:485-499](src/s2a2/iloha_eval.py#L485-L499) and [src/s2a2/iloha_server.py:143-156](src/s2a2/iloha_server.py#L143-L156) |
| RealSense serial numbers and resolutions | `SOUNDREAL_CAMERA_CONFIGS` / `DEFAULT_CAMERA_CONFIGS`, [src/s2a2/soundreal_utils.py:46-80](src/s2a2/soundreal_utils.py#L46-L80) |
| TAMAGO microphone device ids | `TAMAGO_DEVICE_IDS`, [src/s2a2/soundreal_utils.py:86](src/s2a2/soundreal_utils.py#L86) (or the `SOUNDREAL_DEVICE_IDS` env var) |
| Data-collection settings (speaker, sound, episode count) | Module constants at the top of [src/s2a2/iloha_server.py:41-48](src/s2a2/iloha_server.py#L41-L48) |
| WebSocket port for the Quest app | `self.websocket_port`, [src/s2a2/iloha_server.py:60](src/s2a2/iloha_server.py#L60) |

## Data collection with the Meta Quest 3

### 1. Install the Quest app

Build and install [KyotoVLATech/AlohaController](https://github.com/KyotoVLATech/AlohaController)
onto the Meta Quest 3. Put the headset and the PC on the same local network.

### 2. Configure the collection run

`iloha_server.py` takes **no command-line arguments**. All settings are module
constants at the top of the file ([src/s2a2/iloha_server.py:41-48](src/s2a2/iloha_server.py#L41-L48)) —
edit them before starting the server:

```python
USE_RIGHT_SPEAKER = True                # which speaker plays the sound
FIXED_SOUND_INDEX: Optional[int] = 1    # 0 = buzzer, 1 = piyopiyo, None = first half 0 / second half 1
IS_SOUND_SHAKE = True                   # True = exploratory (shake) task: no playback, weaker grip
EPISODE_NUM = 50                        # the server shuts down after this many saved episodes
```

With `IS_SOUND_SHAKE = True` the recorded task string becomes
`"Shake the cans. Pick up the one that makes sound and place it in the box."`;
otherwise it is the `soundReal-m4-f10-s2-p0` task description.

### 3. Start the server

```bash
uv run src/s2a2/iloha_server.py
```

The server listens for WebSocket connections on `0.0.0.0:8080`. During the handshake the
Quest app sends a `joint_send_port`, and the server then opens a **UDP** socket on that port
for the joint stream. If the PC runs a firewall, allow both:

```bash
sudo ufw allow 8080/tcp
sudo ufw allow 9000:9100/udp   # widen to cover the UDP port your Quest app reports
```

### 4. Record

Press **Connect** in the Quest app. Once the status turns to `connected`, the robot follows
the Quest controllers. Recording starts automatically on the first received action and runs
at 30 FPS (the control loop itself runs at 60 Hz). Use the app buttons to send
`recording`, `save_data`, `discard_data`, and `reset_robot`.

For the first 3 seconds of each episode, and whenever a commanded joint jumps by more than
0.2 rad, commands are sent in relative mode to avoid a sudden lurch.

Episodes are written to `datasets/soundReal-m4-f10-s2-p0_<N>`, where `<N>` is the next unused
index. After `EPISODE_NUM` saved episodes the dataset is finalized and the robot shuts down.

## Training on real-robot data

Real-robot datasets train exactly like simulated ones. Only `act` and `diffusion` can be
evaluated on the real robot ([src/s2a2/iloha_eval.py:50](src/s2a2/iloha_eval.py#L50)), so
train one of those:

```bash
uv run lerobot-train \
  --dataset.repo_id=local/soundReal-m4-f10-s2-p0 \
  --dataset.root=datasets/soundReal-m4-f10-s2-p0 \
  --policy.type=act \
  --output_dir=outputs/train/act_soundReal-m4-f10-s2-p0_seed0 \
  --job_name=act_soundReal-m4-f10-s2-p0_seed0 \
  --seed=0 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --batch_size=8 \
  --steps=200000
```

## Evaluation on the real robot

`iloha_eval.py` loads a checkpoint, runs it on the robot, plays the speaker sound, and asks
you on the terminal after each episode whether it succeeded.

### Arguments

| Flag | Default | Meaning |
| --- | --- | --- |
| `--policy_path` | *(required)* | Path to `.../checkpoints/<step>/pretrained_model`. Only `act` and `diffusion` are supported. |
| `--dataset_path` | *(required)* | Training dataset, used for normalization statistics and feature validation. |
| `--output_root` | `datasets` | Where the recorded evaluation dataset is written. |
| `--save_data` | off | Record the evaluation episodes as a new LeRobot dataset. |
| `--episode_time_s` | `60.0` | Length of one episode in seconds. |
| `--num_episodes` | `1` | Number of episodes to run. |
| `--display_data` | off | Live visualization with rerun. |
| `--device` | `cuda` | `cuda` or `cpu`. |
| `--seed` | `0` | Seeds the random sound/speaker sampling. |
| `--sound_index` | random | `0` = `sounds/0.wav` (sound A), `1` = `sounds/1.wav` (sound B). Omit to randomize per episode. |
| `--speaker` | random | `left` or `right`. Omit to randomize per episode. |
| `--is-sound-shake` | off | Exploratory (shake) task: disables playback and lowers the gripper current limit. |
| `--audio_preroll_s` | `0.5` | Delay between starting playback and starting the rollout. |
| `--output_device` | system default | `sounddevice` output device id. |
| `--input_device_ids` | auto | Comma-separated TAMAGO input device ids, e.g. `6,7,8,9`. |
| `--no-remap-soundmap-channels` | remap is **on** | Disables the clockwise-from-top-right SoundMap channel remap. |

### L&I task (sound identifies the target)

```bash
uv run src/s2a2/iloha_eval.py \
  --policy_path outputs/train/act_soundRealAll-m4-f10-s2-p0_seed0/checkpoints/200000/pretrained_model \
  --dataset_path datasets/soundRealAll-m4-f10-s2-p0 \
  --output_root datasets/eval \
  --episode_time_s 15 \
  --num_episodes 25 \
  --save_data \
  --sound_index 0 \
  --speaker right
```

### Exploratory task (the robot must shake the cans to find the sound source)

```bash
uv run src/s2a2/iloha_eval.py \
  --policy_path outputs/train/act_soundRealShake-m4-f10-s2-p0_seed0/checkpoints/200000/pretrained_model \
  --dataset_path datasets/soundRealShake-m4-f10-s2-p0 \
  --output_root datasets/eval \
  --episode_time_s 20 \
  --num_episodes 25 \
  --save_data \
  --is-sound-shake
```

Replace the `--policy_path` and `--dataset_path` values with the checkpoint and dataset you
actually have; the names above are the ones used for the paper experiments.

## Output files

| File | Written when | Contents |
| --- | --- | --- |
| `<output_root>/eval_soundReal-m4-f10-s2-p0_<N>/` | `--save_data` | LeRobot dataset of the evaluation episodes. |
| `sound_conditions.csv` | `--save_data` | Which sound index and speaker were used per episode. |
| `episode_success.csv` | always | Your success/failure answer per episode. Written next to the evaluation dataset, or into `--dataset_path` when `--save_data` is off. |

## Troubleshooting

**`Set SOUNDREAL_DEVICE_IDS or connect the arrays.`**
Fewer than four TAMAGO arrays were found. Check `sounddevice.query_devices()` and export
`SOUNDREAL_DEVICE_IDS`.

**`Permission denied: '/dev/ttyUSB0'`**
Add your user to the `dialout` group and log back in.

**`No such file or directory: '/dev/ttyUSB_RightDynamixel'`**
The udev symlink is missing. See [Serial ports](#serial-ports).

**No RealSense frames, or frames rejected as stale**
Frames older than 80 ms are discarded. Confirm both cameras are on USB 3 and that their
serial numbers match `soundreal_utils.py`. Verify with `rs-enumerate-devices` from
`librealsense`.

**The Quest app connects but the robot does not move**
Only UDP packets with `mode == 1` drive the robot. Confirm the app is in teleoperation mode
and that the UDP port it reported is not blocked by the firewall.
