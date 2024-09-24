#/bin/bash

# this script assumes running on M1 Mac
export DOCKER_DEFAULT_PLATFORM=linux/amd64

DOCKER_IMAGE=garrettgoon/rag-pdf-base

docker buildx build -f Dockerfile_base --platform linux/amd64 . -t $DOCKER_IMAGE

echo "Running  docker push $DOCKER_IMAGE..."

docker push $DOCKER_IMAGE
