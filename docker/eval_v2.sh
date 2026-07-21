#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_WORKDIR="/workspace/myproject"
IMAGE_TAG="${IMAGE_TAG:-myproject:latest}"

usage() {
    cat <<'EOF'
Usage:
  ./docker/eval_v2.sh --gpu <list>
  ./docker/eval_v2.sh <gpu>

Required:
  --gpu <list>  CUDA_VISIBLE_DEVICES inside container (e.g. 0 or 1,2)

Options:
  -h, --help   Show this help
EOF
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

inside_main() {
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        usage
        exit 0
    fi

    cd "${ROOT_DIR}"
    mkdir -p outputs/eval_v2 outputs/train

    uv run src/eval_policy_v2.py
}

host_main() {
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        usage
        exit 0
    fi

    CONTAINER_CUDA_VISIBLE_DEVICES=""
    POSITIONAL_INDEX=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --gpu)
                if [[ $# -lt 2 ]]; then
                    echo "--gpu requires a value." >&2
                    usage >&2
                    exit 1
                fi
                CONTAINER_CUDA_VISIBLE_DEVICES="$2"
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
                if [[ ${POSITIONAL_INDEX} -eq 1 ]]; then
                    CONTAINER_CUDA_VISIBLE_DEVICES="${CONTAINER_CUDA_VISIBLE_DEVICES:-$1}"
                else
                    echo "Unexpected positional argument: $1" >&2
                    usage >&2
                    exit 1
                fi
                shift
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

    docker "${DOCKER_ARGS[@]}" "${IMAGE_TAG}" bash docker/eval_v2.sh "__inside__"
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
