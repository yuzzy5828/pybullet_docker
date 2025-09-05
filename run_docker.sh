docker run -it --rm \
    --env="DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="/${HOME}/.Xauthority:/root/.Xauthority:rw" \
    --volume="/${HOME}/docker/pybullet_docker/workdir:/srv/workdir:rw" \
    --gpus all \
    --network host \
    --name pybullet_simulation \
    pybullet:latest