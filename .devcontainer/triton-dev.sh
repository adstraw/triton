#!/bin/bash
# Start and enter Triton dev container
set -e

CONTAINER_NAME="triton-dev"
IMAGE_NAME="triton-devcontainer"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRITON_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Build image if it doesn't exist
if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "Building ${IMAGE_NAME}..."
    docker build -t "${IMAGE_NAME}" -f "${SCRIPT_DIR}/Dockerfile" "${TRITON_DIR}"
fi

# Start container if not running
if ! docker ps --filter "name=${CONTAINER_NAME}" --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "Starting ${CONTAINER_NAME}..."
    docker run -d --name "${CONTAINER_NAME}" \
        --gpus=all \
        --ipc=host \
        --cap-add=SYS_PTRACE \
        --security-opt=seccomp=unconfined \
        -v "${TRITON_DIR}:/triton" \
        -w /triton \
        "${IMAGE_NAME}" sleep infinity
    
    docker exec "${CONTAINER_NAME}" git config --global --add safe.directory /triton
fi

echo "Entering ${CONTAINER_NAME}..."
docker exec -it "${CONTAINER_NAME}" bash
