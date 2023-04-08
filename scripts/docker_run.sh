xhost +
nvidia-docker run -it \
    --name bevfusion \
    -u $(id -u):$(id -g) \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/robot/dataset:/dataset \
    -v `pwd`:/BEV-RF \
    --net host \
    --shm-size 16g \
    bevfusion:dev /bin/bash

# passwd: robot
