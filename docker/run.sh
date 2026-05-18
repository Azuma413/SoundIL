#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_WORKDIR="/workspace/myproject"
IMAGE_TAG="${IMAGE_TAG:-myproject:latest}"

usage() {
    cat <<'EOF'
Usage:
  ./docker/run.sh login
  ./docker/run.sh login [options]
  ./docker/run.sh train --dataset-name <name> [options]
  ./docker/run.sh eval --training-name <name> [options]
  ./docker/run.sh train-eval --dataset-name <name> [options]

Common options:
  --gpu <list>             CUDA_VISIBLE_DEVICES inside container (e.g. 0 or 1,2)
  --dataset-name <name>     Dataset directory under datasets/
  --training-name <name>    Directory under outputs/train/ (required for eval)
  --policy <name>           act | diffusion | vqbet | pi0
  --steps <int>             Training steps
  --save-freq <int>         Checkpoint save frequency
  --seeds "<list>"          Space or comma separated seed list
  --batch-size <int>        Override batch size
  --policy-device <name>    Usually cuda
  --episode-num <int>       Number of evaluation episodes
  --checkpoint-step <step>  Evaluation checkpoint, default: steps
  --show-viewer             Enable Genesis viewer during evaluation
  --no-eval                 Skip evaluation after training
  --extra-train-arg <arg>   Additional lerobot-train argument
  --extra-eval-arg <arg>    Additional eval_policy.py argument
  -h, --help                Show this help
EOF
}

default_batch_size() {
    case "$1" in
        pi0) echo 4 ;;
        act) echo 8 ;;
        diffusion|vqbet) echo 32 ;;
        *)
            echo "Unsupported policy: $1" >&2
            exit 1
            ;;
    esac
}

append_policy_args() {
    local policy="$1"
    case "${policy}" in
        diffusion)
            POLICY_ARGS+=(--policy.use_separate_rgb_encoder_per_camera=true)
            ;;
        pi0)
            POLICY_ARGS+=(
                --policy.pretrained_path=lerobot/pi0_base
                --policy.gradient_checkpointing=true
                --policy.dtype=bfloat16
            )
            ;;
        act|vqbet)
            ;;
        *)
            echo "Unsupported policy: ${policy}" >&2
            exit 1
            ;;
    esac
}

run_login() {
    uv run huggingface-cli login
    uv run wandb login
}

run_eval() {
    local training_name="$1"

    CMD=(
        uv run src/eval_policy.py
        --training-name "${training_name}"
        --checkpoint-step "${CHECKPOINT_STEP}"
        --episode-num "${EPISODE_NUM}"
        --observation-height "${OBSERVATION_HEIGHT}"
        --observation-width "${OBSERVATION_WIDTH}"
    )

    if [[ -n "${DATASET_NAME}" ]]; then
        CMD+=(--dataset-name "${DATASET_NAME}")
    fi

    if [[ "${SHOW_VIEWER}" == "true" ]]; then
        CMD+=(--show-viewer)
    fi

    CMD+=("${EXTRA_EVAL_ARGS[@]}")
    "${CMD[@]}"
}

run_train() {
    local seed="$1"
    local training_name="${POLICY}_${DATASET_NAME}_seed${seed}"

    CMD=(
        uv run lerobot-train
        --dataset.repo_id="local/${DATASET_NAME}"
        --dataset.root="datasets/${DATASET_NAME}"
        --policy.type="${POLICY}"
        --output_dir="outputs/train/${training_name}"
        --job_name="${training_name}"
        --policy.device="${POLICY_DEVICE}"
        --policy.push_to_hub=false
        --wandb.enable="${WANDB_ENABLE}"
        --wandb.disable_artifact=true
        --dataset.video_backend="${VIDEO_BACKEND}"
        --batch_size="${BATCH_SIZE}"
        --steps="${STEPS}"
        --save_freq="${SAVE_FREQ}"
        --seed="${seed}"
    )

    CMD+=("${POLICY_ARGS[@]}")
    CMD+=("${EXTRA_TRAIN_ARGS[@]}")

    echo "==> Training ${training_name} (seed=${seed})"
    "${CMD[@]}"

    if [[ "${RUN_EVAL}" == "true" ]]; then
        echo "==> Evaluating ${training_name}"
        run_eval "${training_name}"
    fi
}

inside_main() {
    MODE="${1:-}"
    if [[ -z "${MODE}" || "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
        usage
        exit 0
    fi
    shift || true

    DATASET_NAME="${DATASET_NAME:-}"
    TRAINING_NAME="${TRAINING_NAME:-}"
    POLICY="${POLICY:-act}"
    STEPS="${STEPS:-100000}"
    SAVE_FREQ="${SAVE_FREQ:-10000}"
    SEEDS="${SEEDS:-0 1 2}"
    BATCH_SIZE="${BATCH_SIZE:-}"
    POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
    VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
    WANDB_ENABLE="${WANDB_ENABLE:-true}"
    RUN_EVAL="${RUN_EVAL:-true}"
    EPISODE_NUM="${EPISODE_NUM:-100}"
    CHECKPOINT_STEP="${CHECKPOINT_STEP:-}"
    OBSERVATION_HEIGHT="${OBSERVATION_HEIGHT:-224}"
    OBSERVATION_WIDTH="${OBSERVATION_WIDTH:-224}"
    SHOW_VIEWER="${SHOW_VIEWER:-false}"
    EXTRA_TRAIN_ARGS=()
    EXTRA_EVAL_ARGS=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dataset-name)
                DATASET_NAME="$2"
                shift 2
                ;;
            --training-name)
                TRAINING_NAME="$2"
                shift 2
                ;;
            --policy)
                POLICY="$2"
                shift 2
                ;;
            --steps)
                STEPS="$2"
                shift 2
                ;;
            --save-freq)
                SAVE_FREQ="$2"
                shift 2
                ;;
            --seeds)
                SEEDS="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --policy-device)
                POLICY_DEVICE="$2"
                shift 2
                ;;
            --episode-num)
                EPISODE_NUM="$2"
                shift 2
                ;;
            --checkpoint-step)
                CHECKPOINT_STEP="$2"
                shift 2
                ;;
            --show-viewer)
                SHOW_VIEWER="true"
                shift
                ;;
            --no-eval)
                RUN_EVAL="false"
                shift
                ;;
            --extra-train-arg)
                EXTRA_TRAIN_ARGS+=("$2")
                shift 2
                ;;
            --extra-eval-arg)
                EXTRA_EVAL_ARGS+=("$2")
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown argument: $1" >&2
                usage >&2
                exit 1
                ;;
        esac
    done

    cd "${ROOT_DIR}"
    mkdir -p outputs/train outputs/eval outputs/wandb

    case "${MODE}" in
        login)
            run_login
            ;;
        eval)
            if [[ -z "${TRAINING_NAME}" ]]; then
                echo "--training-name is required for eval." >&2
                usage >&2
                exit 1
            fi

            if [[ -z "${CHECKPOINT_STEP}" ]]; then
                CHECKPOINT_STEP="last"
            fi

            run_eval "${TRAINING_NAME}"
            ;;
        train|train-eval)
            if [[ -z "${DATASET_NAME}" ]]; then
                echo "--dataset-name is required for ${MODE}." >&2
                usage >&2
                exit 1
            fi

            if [[ -z "${CHECKPOINT_STEP}" ]]; then
                CHECKPOINT_STEP="${STEPS}"
            fi

            if [[ "${MODE}" == "train" ]]; then
                RUN_EVAL="false"
            fi

            if [[ -z "${BATCH_SIZE}" ]]; then
                BATCH_SIZE="$(default_batch_size "${POLICY}")"
            fi

            POLICY_ARGS=()
            append_policy_args "${POLICY}"

            IFS=' ' read -r -a SEED_LIST <<< "${SEEDS//,/ }"

            for seed in "${SEED_LIST[@]}"; do
                run_train "${seed}"
            done
            ;;
        *)
            echo "Unknown mode: ${MODE}" >&2
            usage >&2
            exit 1
            ;;
    esac
}

ensure_image() {
    if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
        docker build -t "${IMAGE_TAG}" -f "${ROOT_DIR}/Dockerfile" "${ROOT_DIR}"
    fi
}

prepare_auth_mounts() {
    mkdir -p \
        "${HOME}/.cache/huggingface" \
        "${HOME}/.cache/wandb" \
        "${HOME}/.config/wandb"

    touch "${HOME}/.netrc"
    chmod 600 "${HOME}/.netrc" || true
}

host_main() {
    MODE="${1:-}"
    if [[ -z "${MODE}" || "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
        usage
        exit 0
    fi
    shift || true

    CONTAINER_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    FORWARD_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --gpu)
                CONTAINER_CUDA_VISIBLE_DEVICES="$2"
                shift 2
                ;;
            *)
                FORWARD_ARGS+=("$1")
                shift
                ;;
        esac
    done

    ensure_image
    prepare_auth_mounts

    HOST_UID="$(id -u)"
    HOST_GID="$(id -g)"

    DOCKER_ARGS=(
        run
        --rm
        -it
        --gpus all
        --ipc=host
        --shm-size=8g
        -w "${CONTAINER_WORKDIR}"
        -v "${ROOT_DIR}:${CONTAINER_WORKDIR}"
        -e CUDA_VISIBLE_DEVICES="${CONTAINER_CUDA_VISIBLE_DEVICES}"
        -e WANDB_API_KEY="${WANDB_API_KEY:-}"
        -e WANDB_CONFIG_DIR="/root/.config/wandb"
        -e WANDB_CACHE_DIR="/root/.cache/wandb"
        -e HF_TOKEN="${HF_TOKEN:-}"
        -e HF_HOME="/root/.cache/huggingface"
        -e HUGGINGFACE_HUB_CACHE="/root/.cache/huggingface/hub"
        -v "${HOME}/.cache/huggingface:/root/.cache/huggingface"
        -v "${HOME}/.cache/wandb:/root/.cache/wandb"
        -v "${HOME}/.config/wandb:/root/.config/wandb"
        -v "${HOME}/.netrc:/root/.netrc"
    )

    docker "${DOCKER_ARGS[@]}" "${IMAGE_TAG}" bash docker/run.sh "__inside__" "${MODE}" "${FORWARD_ARGS[@]}"
    STATUS=$?

    # Training/eval runs inside the container as root, so restore host ownership
    # on bind-mounted artifacts before returning control to the caller.
    docker run --rm \
        -v "${ROOT_DIR}:${CONTAINER_WORKDIR}" \
        -w "${CONTAINER_WORKDIR}" \
        "${IMAGE_TAG}" \
        bash -lc "if [[ -d outputs ]]; then chown -R ${HOST_UID}:${HOST_GID} outputs; fi" >/dev/null 2>&1 || true

    exit "${STATUS}"
}

if [[ "${1:-}" == "__inside__" ]]; then
    shift
    inside_main "$@"
else
    host_main "$@"
fi
