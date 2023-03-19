nvidia-docker run -it \
    --name bevfusion \
    -u $(id -u):$(id -g) \
    -v /home/robot/dataset:/dataset \
    -v `pwd`:/BEV-RF \
    --shm-size 16g \
    bevfusion:dev /bin/bash

# passwd: robot
