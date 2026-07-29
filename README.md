# S2A2: Audio-Visual Imitation Learning for Manipulation Tasks Using Acoustic Spatial Information

<p align="center">
  🌐 <a href="https://azuma413.github.io/projects/s2a2"><b>Project Page</b></a>
  &nbsp;|&nbsp;
  📄 <a href="https://arxiv.org/abs/2607.26047"><b>Paper (arXiv)</b></a>
</p>

S2A2 is a research codebase for studying imitation learning policies that use acoustic spatial
cues in addition to visual observations. The project extends LeRobot-based policy training with
simulated and real robot manipulation tasks in which sound can identify the relevant object,
target, or task condition.

The core idea is to make acoustic information available to standard imitation learning policies
such as ACT, Diffusion Policy, VQ-BeT, and Pi0. In simulation, sound propagation and
microphone-array observations are generated with Pyroomacoustics and integrated into Genesis
environments. Policies can then be trained and evaluated with the same LeRobot dataset and
checkpoint structure used by the upstream training tools.

## Contents

- [What This Repository Contains](#what-this-repository-contains)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Naming Conventions](#naming-conventions)
- [Quick Start](#quick-start)
- [Simulation Workflow](#simulation-workflow)
- [Representation Analysis With t-SNE](#representation-analysis-with-t-sne)
- [Real-Robot Workflow](#real-robot-workflow)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Citation](#citation)

## What This Repository Contains

- Simulation environments for manipulation tasks with and without sound.
- Acoustic observation generation using microphone-array processing, sound maps, and
  spectrogram-like inputs.
- Dataset collection scripts that create LeRobot-format datasets from scripted expert behavior.
- Training entry points for ACT, Diffusion Policy, VQ-BeT, and Pi0.
- Evaluation utilities for simulated rollouts, real-robot Iloha experiments, and representation
  analysis with t-SNE.

## Repository Layout

```text
.
├── images/                     Texture assets used by the simulator
├── libs/
│   ├── Genesis/                Genesis submodule (physics simulator)
│   └── lerobot/                LeRobot submodule (policies, datasets, training)
├── sounds/                     Source audio files used by the simulator
│                               0.wav = sound A, 1.wav = sound B, 2.wav = sound C, 3.wav = soundDiff's sound B
├── src/
│   └── s2a2/                   Main Python package
│       ├── env/
│       │   ├── genesis_env.py  Genesis environment wrapper and task-name parser
│       │   └── tasks/
│       │       ├── normal.py       Baseline manipulation tasks without sound
│       │       ├── sound.py        Acoustic-aware manipulation tasks
│       │       └── sound_camera.py Sound simulation and acoustic observation utilities
│       ├── eval_policy.py      Simulation policy evaluation
│       ├── iloha_eval.py       Real-robot policy evaluation
│       ├── iloha_server.py     Real-robot data collection server
│       ├── make_sim_dataset.py Simulation dataset generation
│       ├── plot_tsne.py        t-SNE visualization of policy representations
│       └── soundreal_utils.py  Real-world audio processing utilities
├── URDF/                       URDF assets for simulation
├── pyproject.toml              Python dependency and uv configuration
├── REAL_ROBOT.md               Real-robot data collection and evaluation
└── README.md
```

Directories that the scripts create as they run — `datasets/`, `outputs/` — are not tracked by git.

## Requirements

The setup was verified on a clean Ubuntu 24.04 container with the versions below. Other versions
are likely to work but have not been tested.

| Item | Verified version | Notes |
| --- | --- | --- |
| OS | Ubuntu 24.04 LTS | |
| GPU | CUDA-capable NVIDIA GPU | Required in practice. Genesis is always asked for its GPU backend; if no GPU is visible it prints `Torch GPU backend not available. Falling back to CPU device.` and continues far too slowly to be usable. Training with `--policy.device=cuda` fails outright. |
| NVIDIA driver | 595.71.05 | PyTorch is installed from the CUDA 12.8 wheel index, which needs driver **≥ 525.60.13**. The CUDA toolkit does **not** need to be installed separately; the wheels bundle their own CUDA runtime. |
| Python | 3.10.20 | Pinned by `.python-version`; `uv` downloads it for you, so no system Python 3.10 is needed. |
| uv | 0.11.33 | |
| PyTorch | 2.7.1+cu128 | Resolved by `uv sync`. `uv.lock` is not committed, so the exact resolved versions can differ between machines. |
| Genesis | 0.3.6 | Built from the `libs/Genesis` submodule. |
| LeRobot | 0.4.1 | Built from the `libs/lerobot` submodule. |
| Disk space | ≈ 10 GB | For the virtual environment alone (9.3 GB as measured). Datasets and checkpoints need much more. |

Check your driver with:

```bash
nvidia-smi
```

The `Driver Version` column must be at least `525.60.13`. If `nvidia-smi` is not found, install
the driver first (`sudo ubuntu-drivers install`) and reboot.

## Installation

### 1. System packages

```bash
sudo apt update
sudo apt install -y build-essential git curl ca-certificates portaudio19-dev ffmpeg libglu1-mesa
```

Why each is needed:

- `build-essential` — `pyaudio` and `evdev` have no prebuilt wheels and are compiled from source.
  Without it `uv sync` fails with `error: command 'cc' failed: No such file or directory` and
  `The 'linux/input.h' and 'linux/input-event-codes.h' include files are missing`.
- `git` — cloning the repository and its submodules.
- `curl`, `ca-certificates` — downloading the `uv` installer.
- `portaudio19-dev` — headers required to build `pyaudio`.
- `ffmpeg` — video encoding and decoding for LeRobot datasets.
- `libglu1-mesa` — Genesis loads `pygel3d` at import time, which needs `libGLU.so.1`. Without it
  every script fails at startup with
  `OSError: libGLU.so.1: cannot open shared object file`.

A desktop install of Ubuntu usually already has `libglu1-mesa`; a server or container install does
not. Installing it when it is already present is harmless.

### 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

`source` makes `uv` available in the current shell; new shells pick it up automatically. Confirm
the install:

```bash
uv --version
```

### 3. Clone the repository

The two submodules are mandatory — `libs/lerobot` and `libs/Genesis` are installed as editable
local packages, so `uv sync` fails if they are empty.

```bash
git clone --recurse-submodules https://github.com/Azuma413/SoundIL.git
cd SoundIL
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 4. Create the environment

```bash
uv sync
```

This downloads roughly 10 GB of wheels (PyTorch with CUDA 12.8, Genesis, LeRobot and their
dependencies) and takes 10–30 minutes on a first run.

### 5. Verify the installation

```bash
uv run python -c "import torch, genesis, lerobot, pyroomacoustics; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
```

`cuda available: True` means the GPU is visible to PyTorch. If it prints `False`, the simulation
scripts will fail — check the driver version before continuing.

> **Run every command from the repository root.** The scripts resolve `datasets/`, `outputs/`,
> `sounds/`, `images/wood.jpg`, and `URDF/box/box.urdf` relative to the current directory.

## Naming Conventions

Three related names appear throughout the workflow. Understanding them makes the commands below
self-explanatory.

### Task name

A task name has the form `<task-type>-m<M>-f<F>-s<S>-p<P>`, for example `soundSim-m4-f10-s2-p0`.
It is parsed in [src/s2a2/env/genesis_env.py:9-64](src/s2a2/env/genesis_env.py#L9-L64).

**`<task-type>`** — what the robot has to do:

| Task type | Description |
| --- | --- |
| `normal` | No sound. Pick the cube of a given color from a red/blue/green set and place it in the box. |
| `normal-fix` | No sound. Always pick the red cube. |
| `sound` | Two identical-looking speakers; pick the one that is making sound. |
| `soundDiff` | One speaker; place it in the right box for sound A, the left box for sound B. |
| `soundShake` | Two identical-looking speakers, silent until moved; pick the one that makes sound when shaken. |
| `soundAll` | Two speakers; pick the one playing sound A, then place it right for sound B or left for sound C. |
| `soundSim` | Two speakers, one playing sound A or B; pick the sounding one and place it right for A, left for B. |
| `soundReal` | The real-robot task. See [REAL_ROBOT.md](REAL_ROBOT.md). |

**`m<M>`** — number of simulated microphone arrays. All released datasets use `m4`.

**`f<F>`** — how often acoustic observations are recomputed, in Hz. The simulation runs at 30 FPS,
so `f10` recomputes every 3 frames and `f30` every frame.

**`s<S>`** — which observations the policy receives:

| Value | Observations |
| --- | --- |
| `s0` | Camera images only (no microphones at all) |
| `s1` | Camera images + sound map |
| `s2` | Camera images + sound map + spectrogram |
| `s3` | Camera images + spectrogram |

**`p<P>`** — post-processing applied to the acoustic images:

| Value | Processing |
| --- | --- |
| `p0` | Raw |
| `p1` | Gaussian filter |
| `p2` | Temporal smoothing |
| `p3` | Gaussian filter + temporal smoothing |
| `p4` | Feature transform (marker image instead of the raw map) |

### Dataset name

**A dataset name is a task name plus an auto-incremented index**, joined with an underscore:
`<task-name>_<index>`. `make_sim_dataset.py` writes to `datasets/<task-name>_<index>` and picks
the lowest index not already in use, so generating a dataset for `soundSim-m4-f10-s2-p0` the first
time produces:

```text
datasets/soundSim-m4-f10-s2-p0_0
```

and the dataset name to pass to the training and evaluation commands is
`soundSim-m4-f10-s2-p0_0`. Running the generator again would create `..._1`, and so on. The index
exists so that repeated collections of the same task do not overwrite each other.

### Training name

The training name is the directory under `outputs/train/`. This repository uses the convention:

```text
<policy>_<dataset-name>_seed<N>
```

for example `act_soundSim-m4-f10-s2-p0_0_seed0`. The evaluation scripts rely on this format: they
take the leading segment as the policy type, strip the trailing `seed<N>` segment, and treat what
remains as the dataset name ([src/s2a2/eval_policy.py:118-142](src/s2a2/eval_policy.py#L118-L142)).
Deviating from it means you have to pass `--dataset-name` explicitly.

## Quick Start

The commands below run end to end and produce a trained ACT policy on the `soundSim` task with a
success rate report. They are literal — nothing needs to be substituted.

```bash
# 1. Generate a dataset (default task: soundSim-m4-f10-s2-p0, 100 episodes)
uv run src/s2a2/make_sim_dataset.py

# 2. Train ACT on it
uv run lerobot-train \
  --dataset.repo_id=local/soundSim-m4-f10-s2-p0_0 \
  --dataset.root=datasets/soundSim-m4-f10-s2-p0_0 \
  --policy.type=act \
  --output_dir=outputs/train/act_soundSim-m4-f10-s2-p0_0_seed0 \
  --job_name=act_soundSim-m4-f10-s2-p0_0_seed0 \
  --seed=0 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --batch_size=8 \
  --steps=100000

# 3. Evaluate the final checkpoint over 100 simulated episodes
uv run src/s2a2/eval_policy.py \
  --training-name act_soundSim-m4-f10-s2-p0_0_seed0 \
  --dataset-name soundSim-m4-f10-s2-p0_0 \
  --checkpoint-step 100000 \
  --episode-num 100
```

The success rate is printed and also written to
`outputs/eval/act_soundSim-m4-f10-s2-p0_0_seed0_100000/success_rate.txt`.

These steps take hours, not minutes. As a reference point, on the machine used for the paper
experiments (a single NVIDIA GeForce RTX 5090) dataset generation ran at roughly 30 episodes per
hour, and ACT training reached 100,000 steps at `--batch_size=8` in about 1.5 hours. Consider
running the first two steps under `tmux` or `nohup`.

## Simulation Workflow

### Generate a Dataset

Simulation datasets are produced by a scripted expert policy and saved under `datasets/` in
LeRobot Dataset V3 format.

```bash
uv run src/s2a2/make_sim_dataset.py
```

**This script takes no command-line arguments.** To generate a different task, edit the
`task_candidates` list in [src/s2a2/make_sim_dataset.py:324-326](src/s2a2/make_sim_dataset.py#L324-L326):

```python
task_candidates = [
    "soundSim-m4-f10-s2-p0",
]
```

Listing several entries generates one dataset per task in sequence. The other settings are
arguments to `main(...)` at the bottom of the same file
([src/s2a2/make_sim_dataset.py:342-351](src/s2a2/make_sim_dataset.py#L342-L351)):

| Setting | Default | Meaning |
| --- | --- | --- |
| `episode_num` | `100` | Episodes to record. Failed episodes are retried and not counted. |
| `observation_height` / `observation_width` | `224` | Rendered image size. |
| `show_viewer` | `False` | Set `True` to watch the simulation in a window (needs a display). |

Episodes are balanced across task variants — for `soundSim`, across both target cubes and both
sound types — so a run of 100 episodes contains 25 of each combination.

The resulting directory is `datasets/<task-name>_<index>`; see
[Dataset name](#dataset-name) for how the index is chosen.

### Train a Policy

Supported policies are `act`, `diffusion`, `vqbet`, and `pi0`. This example trains ACT on the
dataset produced above:

```bash
uv run lerobot-train \
  --dataset.repo_id=local/soundSim-m4-f10-s2-p0_0 \
  --dataset.root=datasets/soundSim-m4-f10-s2-p0_0 \
  --policy.type=act \
  --output_dir=outputs/train/act_soundSim-m4-f10-s2-p0_0_seed0 \
  --job_name=act_soundSim-m4-f10-s2-p0_0_seed0 \
  --seed=0 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --batch_size=8 \
  --steps=100000
```

To train a different policy or dataset, change `--policy.type` and the three names. Keeping the
`<policy>_<dataset-name>_seed<N>` convention for `--output_dir` and `--job_name` lets the
evaluation scripts infer everything else.

To log to Weights & Biases, run `uv run wandb login` once and then swap the flag:

```bash
  --wandb.enable=true --wandb.disable_artifact=true
```

Checkpoints are written to `outputs/train/<training-name>/checkpoints/<step>/pretrained_model`.

### Evaluate a Checkpoint

[src/s2a2/eval_policy.py](src/s2a2/eval_policy.py) loads a trained policy, rebuilds the matching
Genesis task, and runs simulated evaluation episodes:

```bash
uv run src/s2a2/eval_policy.py \
  --training-name act_soundSim-m4-f10-s2-p0_0_seed0 \
  --dataset-name soundSim-m4-f10-s2-p0_0 \
  --checkpoint-step 100000 \
  --episode-num 100
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--training-name` | `act_soundDiff-m4-f10-s2-p0_0` | Directory under `outputs/train/`. Its first segment selects the policy class. |
| `--dataset-name` | inferred from `--training-name` | Directory under `datasets/`, used for normalization statistics and the observation schema. |
| `--checkpoint-step` | `100000` | Checkpoint to load. Numbers are zero-padded to six digits; `last` also works. |
| `--episode-num` | `100` | Number of evaluation episodes. |
| `--observation-height` / `--observation-width` | `224` | Must match the dataset. |
| `--show-viewer` | off | Open the Genesis viewer window. |

Results go to `outputs/eval/<training-name>_<checkpoint-step>/`:

- `success_rate.txt` — success rate plus action statistics.
- `rollout_ep<N>.mp4` — one video per episode, tiled as front view, side view, sound map, and
  spectrogram.

Because the dataset name determines the environment, you can evaluate a policy under a different
acoustic condition than it was trained on by pointing `--dataset-name` at another dataset with
the same observation schema.

## Representation Analysis With t-SNE

[src/s2a2/plot_tsne.py](src/s2a2/plot_tsne.py) evaluates a checkpoint, extracts hidden
representations from inside the policy, and projects them with t-SNE.

```bash
uv run src/s2a2/plot_tsne.py \
  --training-name act_soundSim-m4-f10-s2-p0_0_seed0 \
  --dataset-name soundSim-m4-f10-s2-p0_0 \
  --checkpoint-step 100000 \
  --episode-num 100 \
  --hidden-reduction mean
```

`--color-by` selects how points are colored: `sound_type`, `sound_coordinate`, `success`, or
`episode_step`. If it is omitted, every coloring is generated. Note that `sound_type` labels only
exist for the `soundDiff`, `soundAll`, and `soundSim` tasks; the others record `Unknown`.

`--hidden-reduction` controls how a temporal hidden state becomes t-SNE points:

- `auto` — use the model-specific current-step representation
- `none` — split hidden states by step and treat each step as its own point
- `first` — use the first hidden step
- `last` — use the last hidden step
- `mean` — average hidden states over the step dimension

By default the script hooks a policy-specific layer (`--hidden-layer auto`) plus an intermediate
layer (`--intermediate-hidden-layer auto`), producing two sets of plots. Pass
`--intermediate-hidden-layer none` to skip the second one, or a dotted module path such as
`model.encoder` to choose your own.

Output goes to `outputs/tsne/<training-name>_<checkpoint-step>/`: `tsne_<color-by>.png`,
`hidden_states.npz`, `tsne_metadata.csv`, and `summary.txt`, with `intermediate_`-prefixed
counterparts for the second layer.

## Real-Robot Workflow

Real-robot data collection and evaluation use the Iloha arm, TAMAGO microphone arrays, and a
Meta Quest 3 controller interface. Because that setup needs specific hardware and hardcoded
device names, it is documented separately:

**→ [REAL_ROBOT.md](REAL_ROBOT.md)**

## Troubleshooting

**`error: command 'cc' failed: No such file or directory` during `uv sync`**
`build-essential` is missing. See [System packages](#1-system-packages).

**`The 'linux/input.h' and 'linux/input-event-codes.h' include files are missing` during `uv sync`**
Same cause — `build-essential` pulls in `linux-libc-dev`, which provides these headers.

**`fatal error: portaudio.h: No such file or directory`**
`portaudio19-dev` is missing.

**`OSError: libGLU.so.1: cannot open shared object file` when any script starts**
`libglu1-mesa` is missing. Genesis loads `pygel3d` on import, which links against it.

**`uv sync` fails complaining about `libs/lerobot` or `libs/Genesis`**
The submodules were not checked out, so those directories are empty. Run
`git submodule update --init --recursive` and try again.

**`torch.cuda.is_available()` returns `False`**
The NVIDIA driver is older than 525.60.13, not loaded, or `nvidia-smi` reports a
`Driver/library version mismatch` — that last one means the driver was updated without a reboot.

**`[Genesis] [WARNING] Torch GPU backend not available. Falling back to CPU device.`**
Same cause. The simulation still starts, so it is easy to miss, but it will be far too slow to
finish a dataset. Fix the GPU before leaving a long run unattended.

**`FileNotFoundError` for `images/wood.jpg`, `sounds/1.wav`, or `URDF/box/box.urdf`**
You are not in the repository root. All paths in the code are relative.

**Genesis fails to open a window**
Only `--show-viewer` needs a display. Leave it off for headless machines.

**Evaluation reports a checkpoint that does not exist**
`--checkpoint-step` is zero-padded to six digits, so `100000` maps to
`outputs/train/<training-name>/checkpoints/100000/`. Check what is actually there with
`ls outputs/train/<training-name>/checkpoints/`.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for the full text.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{hiratsuka2026s2a2audiovisualimitationlearning,
      title={S2A2: Audio-Visual Imitation Learning for Manipulation Tasks Using Acoustic Spatial Information}, 
      author={Kaneyoshi Hiratsuka and Benjamin Yen and Ryosuke Kojima},
      year={2026},
      eprint={2607.26047},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2607.26047}, 
}
```
