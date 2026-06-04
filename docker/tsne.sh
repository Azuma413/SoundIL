#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_WORKDIR="/workspace/myproject"
IMAGE_TAG="${IMAGE_TAG:-myproject:latest}"

usage() {
    cat <<'EOF'
Usage:
  ./docker/tsne.sh --policy <name> --dataset-name <name> --gpu <list> [options]
  ./docker/tsne.sh <policy> <dataset-name> <gpu> [options]

Required:
  --policy <name>        act | diffusion | vqbet | pi0
  --dataset-name <name>  Dataset/training dataset name
  --gpu <list>           CUDA_VISIBLE_DEVICES inside container (e.g. 0 or 1,2)

Options:
  --training-name <name>       Override training name
  --checkpoint-step <step>     Default: last
  --episode-num <int>          Default: 100
  --hidden-reduction <mode>    Default: mean
  --extra-tsne-arg <arg>       Additional plot_tsne.py argument
  -h, --help                   Show this help
EOF
}

ensure_image() {
    if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
        docker build -t "${IMAGE_TAG}" -f "${ROOT_DIR}/docker/Dockerfile" "${ROOT_DIR}"
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

inside_main() {
    POLICY=""
    DATASET_NAME=""
    TRAINING_NAME=""
    CHECKPOINT_STEP="last"
    EPISODE_NUM="100"
    HIDDEN_REDUCTION="mean"
    EXTRA_TSNE_ARGS=()
    POSITIONAL_ARGS=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --policy)
                POLICY="$2"
                shift 2
                ;;
            --dataset-name)
                DATASET_NAME="$2"
                shift 2
                ;;
            --training-name)
                TRAINING_NAME="$2"
                shift 2
                ;;
            --checkpoint-step)
                CHECKPOINT_STEP="$2"
                shift 2
                ;;
            --episode-num)
                EPISODE_NUM="$2"
                shift 2
                ;;
            --hidden-reduction)
                HIDDEN_REDUCTION="$2"
                shift 2
                ;;
            --extra-tsne-arg)
                EXTRA_TSNE_ARGS+=("$2")
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            --*)
                echo "Unknown argument: $1" >&2
                usage >&2
                exit 1
                ;;
            *)
                POSITIONAL_ARGS+=("$1")
                shift
                ;;
        esac
    done

    if [[ ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
        POLICY="${POLICY:-${POSITIONAL_ARGS[0]}}"
    fi
    if [[ ${#POSITIONAL_ARGS[@]} -gt 1 ]]; then
        DATASET_NAME="${DATASET_NAME:-${POSITIONAL_ARGS[1]}}"
    fi
    if [[ ${#POSITIONAL_ARGS[@]} -gt 2 ]]; then
        echo "GPU must be passed on the host side; use --gpu or the third positional argument before entering Docker." >&2
        exit 1
    fi

    if [[ -z "${POLICY}" ]]; then
        echo "--policy is required." >&2
        usage >&2
        exit 1
    fi
    if [[ -z "${DATASET_NAME}" ]]; then
        echo "--dataset-name is required." >&2
        usage >&2
        exit 1
    fi

    if [[ -z "${TRAINING_NAME}" ]]; then
        TRAINING_NAME="${POLICY}_${DATASET_NAME}_0_seed0"
    fi

    cd "${ROOT_DIR}"
    mkdir -p outputs/eval outputs/train

    CMD=(
        uv run src/plot_tsne.py
        --training-name "${TRAINING_NAME}"
        --checkpoint-step "${CHECKPOINT_STEP}"
        --dataset-name "${DATASET_NAME}_0"
        --episode-num "${EPISODE_NUM}"
        --hidden-reduction "${HIDDEN_REDUCTION}"
    )

    CMD+=("${EXTRA_TSNE_ARGS[@]}")

    echo "==> Plotting t-SNE for ${TRAINING_NAME}"
    "${CMD[@]}"
}

host_main() {
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        usage
        exit 0
    fi

    CONTAINER_CUDA_VISIBLE_DEVICES=""
    FORWARD_ARGS=()
    POSITIONAL_INDEX=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --gpu)
                CONTAINER_CUDA_VISIBLE_DEVICES="$2"
                shift 2
                ;;
            --policy|--dataset-name|--training-name|--checkpoint-step|--episode-num|--hidden-reduction|--extra-tsne-arg)
                FORWARD_ARGS+=("$1" "$2")
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            --*)
                echo "Unknown argument: $1" >&2
                usage >&2
                exit 1
                ;;
            *)
                POSITIONAL_INDEX=$((POSITIONAL_INDEX + 1))
                if [[ ${POSITIONAL_INDEX} -eq 3 ]]; then
                    CONTAINER_CUDA_VISIBLE_DEVICES="${CONTAINER_CUDA_VISIBLE_DEVICES:-$1}"
                    shift
                else
                    FORWARD_ARGS+=("$1")
                    shift
                fi
                ;;
        esac
    done

    if [[ -z "${CONTAINER_CUDA_VISIBLE_DEVICES}" ]]; then
        echo "--gpu is required." >&2
        usage >&2
        exit 1
    fi

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

    docker "${DOCKER_ARGS[@]}" "${IMAGE_TAG}" bash docker/tsne.sh "__inside__" "${FORWARD_ARGS[@]}"
    STATUS=$?

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
