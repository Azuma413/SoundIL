# Docker Usage

This guide describes the Docker-based workflow for SoundIL. The wrapper scripts build the image when needed, run the project with GPU access, and mount the repository so that datasets and outputs remain on the host machine.

## Prerequisites

- An NVIDIA GPU
- Docker
- NVIDIA Container Toolkit
- A cloned SoundIL repository with submodules

The Docker scripts use the host repository directly. In particular, `datasets/` and `outputs/` are bind-mounted from the host, so generated datasets, checkpoints, evaluation videos, and plots stay available after the container exits.

The default image tag is `myproject:latest`. You can override it with `IMAGE_TAG`:

```bash
IMAGE_TAG=soundil:latest ./docker/run.sh --help
```

## Authentication

Run the login command once before training if you use Hugging Face gated models or Weights & Biases logging:

```bash
./docker/run.sh login
```

The script creates and mounts these host-side paths:

- `${HOME}/.cache/huggingface`
- `${HOME}/.cache/wandb`
- `${HOME}/.config/wandb`
- `${HOME}/.netrc`

The saved credentials are reused by later `train`, `train-eval`, `eval`, and `tsne` runs.

`pi0` uses the gated `google/paligemma-3b-pt-224` model through `lerobot/pi0_base`. Make sure your Hugging Face account has access before running Pi0 experiments.

## Training and Evaluation

Use `train-eval` to train one or more seeds and evaluate each run immediately after training:

```bash
./docker/run.sh train-eval \
  --dataset-name sound-m4-f10-s2-p0_0 \
  --gpu 0 \
  --policy act \
  --seeds 0 \
  --steps 100000 \
  --save-freq 10000
```

The output directory is always:

```text
outputs/train/<policy>_<dataset-name>_seed<seed>
```

For the command above, the training run is saved as:

```text
outputs/train/act_sound-m4-f10-s2-p0_0_seed0
```

## Training Only

Use `train` when you do not want evaluation to run after training:

```bash
./docker/run.sh train \
  --dataset-name sound-m4-f10-s2-p0_0 \
  --gpu 1 \
  --policy vqbet \
  --seeds "0,1,2" \
  --steps 100000 \
  --save-freq 10000
```

If `--batch-size` is omitted, `docker/run.sh` chooses a default by policy:

- `act`: 8
- `diffusion`: 32
- `vqbet`: 32
- `pi0`: 4

## Evaluation Only

Use `eval` to evaluate an existing training run:

```bash
./docker/run.sh eval \
  --training-name diffusion_soundShake-m4-f10-s2-p0_0_seed2 \
  --gpu 0 \
  --dataset-name soundShake-m4-f10-s2-p0_0 \
  --checkpoint-step 100000 \
  --episode-num 100
```

If `--checkpoint-step` is omitted during `eval`, the script uses `last`. During `train-eval`, the default evaluation checkpoint is the value of `--steps`.

To sweep multiple checkpoints:

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

## Common Options

`docker/run.sh` accepts these options for training and evaluation:

- `--gpu`: value for `CUDA_VISIBLE_DEVICES` inside the container, for example `0` or `1,2`
- `--dataset-name`: dataset directory under `datasets/`
- `--training-name`: training directory under `outputs/train/`, required for `eval`
- `--policy`: `act`, `diffusion`, `vqbet`, or `pi0`
- `--steps`: number of training steps
- `--save-freq`: checkpoint save frequency
- `--seeds`: space-separated or comma-separated seed list
- `--batch-size`: override the policy-specific default batch size
- `--policy-device`: usually `cuda`
- `--episode-num`: number of evaluation episodes
- `--checkpoint-step`: checkpoint used for evaluation
- `--show-viewer`: show the Genesis viewer during evaluation
- `--no-eval`: skip evaluation after training
- `--extra-train-arg`: pass one additional argument to `lerobot-train`
- `--extra-eval-arg`: pass one additional argument to `src/eval_policy.py`

For example, to pass additional LeRobot training arguments:

```bash
./docker/run.sh train \
  --dataset-name sound-m4-f10-s2-p0_0 \
  --gpu 0 \
  --policy act \
  --extra-train-arg --wandb.enable=false \
  --extra-train-arg --policy.n_action_steps=8
```

## t-SNE

Use `docker/tsne.sh` to run `src/plot_tsne.py` inside the same Docker environment:

```bash
./docker/tsne.sh \
  --policy diffusion \
  --dataset-name soundDiff-m4-f10-s2-p0 \
  --gpu 2 \
  --checkpoint-step last \
  --episode-num 100 \
  --hidden-reduction mean
```

The script also supports the positional form:

```bash
./docker/tsne.sh diffusion soundDiff-m4-f10-s2-p0 2
```

By default, the wrapper infers:

- `--training-name <policy>_<dataset-name>_0_seed0`
- `--dataset-name <dataset-name>_0` for `src/plot_tsne.py`
- `--checkpoint-step last`
- `--hidden-reduction mean`

Use `--training-name` or `--extra-tsne-arg` when your experiment uses a different naming pattern:

```bash
./docker/tsne.sh \
  --policy act \
  --dataset-name sound-m4-f10-s2-p0 \
  --gpu 0 \
  --training-name act_sound-m4-f10-s2-p0_0_seed1 \
  --extra-tsne-arg --color-by \
  --extra-tsne-arg success
```

## Help

Print the available commands and options:

```bash
./docker/run.sh --help
./docker/tsne.sh --help
```
