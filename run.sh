#!/bin/bash

# Check for GPU on the system
IS_NVIDIA_GPU=$(lspci | grep -i nvidia)
IS_AMD_GPU=$(lspci | grep -i amd)
IMAGE_OVERRIDE=$1 # Optional argument to override default image





# Set IMAGE_NAME based on available GPU
if [ -n "$IS_NVIDIA_GPU" ] || [ '$IMAGE_OVERRIDE' == 'nvidia' ]; then
    IMAGE_NAME=nvcr.io/nvidia/tensorflow:23.12-tf2-py3
elif [ -n "$IS_AMD_GPU" ] || [ '$IMAGE_OVERRIDE' == 'amd' ]; then
    IMAGE_NAME=rocm/tensorflow:latest
else
    IMAGE_NAME=tensorflow/tensorflow:latest
fi

# Export IMAGE_NAME for use in Dockerfile
export IMAGE_NAME

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Please install Docker."
    exit 1
fi

# Check if buildx is available
if docker buildx version &> /dev/null; then
    BUILD_CMD="docker buildx build --build-arg IMAGE_NAME=${IMAGE_NAME} -t mwawrzkow_midi_gan_rnn --load ."
else
    BUILD_CMD="docker build --build-arg IMAGE_NAME=${IMAGE_NAME} -t mwawrzkow_midi_gan_rnn ."
fi

# Build the Docker image
echo "Building Docker image..."
$BUILD_CMD

# Check if build was successful
if [ $? -ne 0 ]; then
    echo "Docker build failed."
    exit 1
fi

# Run the Docker container with appropriate GPU flags
if [ -n "$IS_NVIDIA_GPU" ]; then
    docker run --gpus all -it --rm \
        --shm-size=32g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v ${HOME}/dockerx:/dockerx \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY=${DISPLAY} \
        --net=host \
        mwawrzkow_midi_gan_rnn
elif [ -n "$IS_AMD_GPU" ]; then
    docker run --device=/dev/kfd --device=/dev/dri --group-add video -it --rm \
        --shm-size=32g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v ${HOME}/dockerx:/dockerx \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY=${DISPLAY} \
        --net=host \
        mwawrzkow_midi_gan_rnn
else
    docker run -it --rm \
        --shm-size=32g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v ${HOME}/dockerx:/dockerx \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY=${DISPLAY} \
        --net=host \
        mwawrzkow_midi_gan_rnn
fi
