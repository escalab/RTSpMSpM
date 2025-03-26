#!/bin/bash

GITTOP="$(git rev-parse --show-toplevel 2>&1)"

PROJ="optixspmspm"
DOCKERFILE_PATH="./"
IMAGE_NAME=${PROJ}_image

docker build -t ${IMAGE_NAME} ${DOCKERFILE_PATH}