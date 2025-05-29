#!/usr/bin/env python3
import os

if __name__=="__main__":
    cmd = "docker run \
         --network=host \
         --gpus all \
         -e QT_X11_NO_MITSHM=1 \
         -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
         -v /dev:/dev \
         --device=/dev/video2 \
         --device-cgroup-rule 'c 81:* rmw' \
         --device-cgroup-rule 'c 189:* rmw' \
         -v %s:/host \
         -it msphinx" % (os.path.abspath(os.path.join(os.getcwd(), '..')))
    print(cmd)
    code = os.system(cmd)
