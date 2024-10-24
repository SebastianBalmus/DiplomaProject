#!/bin/bash

DATASET_PATH="/media/DATA/SBALMUS/"
PYTORCH_IMAGE="nvcr.io/nvidia/pytorch"
PYTORCH_TAG="19.08-py3"
IMAGE_NAME="sebastian_pytorch"
CONTAINER_NAME="sebastian_diploma"

# Absolute path of the project
PROJECT_PATH=$(realpath "$(dirname "$(dirname "$0")")")


help () {
    echo ""
    echo
    echo "Syntax:   launch_container [-d|b|h]"
    echo "options:"
    echo "h         Display help"
    echo "d         Absolute path to the dataset"
    echo "b         Use this flag if you want to build the image from scratch"
}

build_image() {
    # If the Pytorch image is not existing on the machine, pull it
    if ! $DOCKER_COMMAND images | grep -q "$PYTORCH_IMAGE"; then
        $DOCKER_COMMAND pull "$IMAGE_NAME:$PYTORCH_TAG"
    fi

    # If there is an existing container, delete it
    if $DOCKER_COMMAND ps -a | grep -q "$CONTAINER_NAME"; then
        $DOCKER_COMMAND stop "$CONTAINER_NAME"
        $DOCKER_COMMAND rm "$CONTAINER_NAME"
    fi

    # If the image already exists, delete it
    if $DOCKER_COMMAND images | grep -q "$IMAGE_NAME"; then
        IMAGE_ID=$($DOCKER_COMMAND images --format "{{.ID}}" "$IMAGE_NAME")
        $DOCKER_COMMAND rmi "$IMAGE_ID"
    fi

    # Rebuild the image
    $DOCKER_COMMAND build -t "$IMAGE_NAME" .
}


# Check the OS
if [[ "$OSTYPE" == "msys" || "$MSYSTEM" == "MINGW32" || "$MSYSTEM" == "MINGW64" ]]; then
    # Windows with MinGW (Git Bash)
    DOCKER_COMMAND="docker"
else
    # Other operating systems (e.g., Linux, macOS)
    DOCKER_COMMAND="sudo docker"
fi


while getopts ":hd:b" option; do
   case $option in
        h) # display Help
            help
            exit;;
        d) # dataset path
            DATASET_PATH=$OPTARG
            ;;
        b) # Build image
            build_image
            ;;
        \?) # Invalid option
            echo "Error: Invalid option"
            Help
            exit;;
   esac
done



# Check the OS
if [[ "$OSTYPE" == "msys" || "$MSYSTEM" == "MINGW32" || "$MSYSTEM" == "MINGW64" ]]; then
    # Windows with MinGW (Git Bash)
    # Start the container
    $DOCKER_COMMAND run \
        --name "$CONTAINER_NAME" \
        --mount src="$PROJECT_PATH",target="/app",type=bind \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -p 8080:8080 \
        -p 8888:8888 \
        -p 6006:6006 \
        -it \
        $IMAGE_NAME
else
    # Start the container
    $DOCKER_COMMAND run \
        --name "$CONTAINER_NAME" \
        --mount src="$PROJECT_PATH",target="/app",type=bind \
        --mount src="$DATASET_PATH",target="/train_path",type=bind \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -p 8080:8080 \
        -p 8888:8888 \
        -p 6006:6006 \
        -it \
        $IMAGE_NAME
fi

