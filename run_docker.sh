#/bin/bash

docker run --shm-size=64g -p 8888:8888 -it --rm --gpus all --name scipostlayouttree \
 --volume $(pwd):/scipostlayouttree_code --volume $(pwd)/../scipostlayout:/scipostlayout nvidia/cuda:12.1.0-devel-ubuntu22.04
# docker run --shm-size=64g -p 8888:8888 -it --rm --gpus all --name scipostlayouttree \
#  --volume $(pwd):/scipostlayouttree_code --volume $(pwd)/../scipostlayout:/scipostlayout scipostlayouttree
