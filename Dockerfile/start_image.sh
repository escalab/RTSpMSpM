#!/bin/bash

GITTOP="$(git rev-parse --show-toplevel 2>&1)"

PROJ="optixspmspm"
IMAGE_NAME=${PROJ}_image
CONTAINER_NAME="${PROJ}_container"
SRC_DIR="${GITTOP}"
SRC_TARGET_DIR="/home/RTSpMSpM"

if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    docker stop ${CONTAINER_NAME}
    docker rm ${CONTAINER_NAME}
fi

docker run -d \
    -it \
    --rm       \
    --gpus all \
    --privileged \
    --ulimit core=-1 \
    --name ${CONTAINER_NAME} \
    -u $(id -u $USER):$(id -g $USER) \
    --mount type=bind,source=${SRC_DIR},target=${SRC_TARGET_DIR} \
    # --mount type=bind,source=/trace,target=/home/trace \
    ${IMAGE_NAME} 