FROM ubuntu:22.04

ENV DEBIAL_FRONTED=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    freeglut3 \
    build-essential \
    x11-apps && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install numpy pybullet

WORKDIR /srv/workdir

CMD ["/bin/bash"]