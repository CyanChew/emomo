#!/usr/bin/env python3
import os

if __name__=="__main__":
    cmd = "xhost +local:root && \
         docker run \
         --network=host \
         --gpus all \
         -e DISPLAY=$DISPLAY \
         -e QT_X11_NO_MITSHM=1 \
         -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
         -v /tmp/.X11-unix:/tmp/.X11-unix \
         -v /dev:/dev \
         --device=/dev/video2 \
         --device-cgroup-rule 'c 81:* rmw' \
         --device-cgroup-rule 'c 189:* rmw' \
         -v %s:/host \
         -it librealsense/librealsense /bin/bash" % (os.path.abspath(os.path.join(os.getcwd(), '..')))
    print(cmd)
    code = os.system(cmd)
