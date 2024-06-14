# Check the OS
if [[ "$OSTYPE" == "msys" || "$MSYSTEM" == "MINGW32" || "$MSYSTEM" == "MINGW64" ]]; then
    # Windows with MinGW (Git Bash)
    DOCKER_COMMAND="docker"
else
    # Other operating systems (e.g., Linux, macOS)
    DOCKER_COMMAND="sudo docker"
fi

cd frontend

$DOCKER_COMMAND rm sebastian_frontend
$DOCKER_COMMAND rmi sebastian_nginx
$DOCKER_COMMAND pull nginx:alpine
$DOCKER_COMMAND build -t sebastian_nginx .

# run Nginx
$DOCKER_COMMAND run \
    --name sebastian_frontend \
    -p 80:80 \
    -it \
    sebastian_nginx
