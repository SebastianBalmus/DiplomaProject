#!/bin/bash

# Check the OS
if [[ "$OSTYPE" == "msys" || "$MSYSTEM" == "MINGW32" || "$MSYSTEM" == "MINGW64" ]]; then
    # Windows with MinGW (Git Bash)
    DOCKER_COMMAND="docker"
else
    # Other operating systems (e.g., Linux, macOS)
    DOCKER_COMMAND="sudo docker"
fi

cd frontend

# Remove the container if it exists
$DOCKER_COMMAND rm -f sebastian_frontend || true

# Remove the image if it exists
$DOCKER_COMMAND rmi -f sebastian_nginx || true

# Pull the base Nginx image
$DOCKER_COMMAND pull nginx:alpine

# Build the Docker image
$DOCKER_COMMAND build -t sebastian_nginx .

# Run the container
$DOCKER_COMMAND run \
    --name sebastian_frontend \
    -p 80:80 \
    -d \
    sebastian_nginx