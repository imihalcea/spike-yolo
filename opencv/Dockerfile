FROM ubuntu:jammy

RUN apt update -y
RUN \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get -y install \
    wget \
    unzip \
    git \
    cmake \
    pkgconf \
    build-essential \
    ninja-build \
    crossbuild-essential-armhf \
    crossbuild-essential-arm64 \
    python3-dev \
    python3-pip \
    python3-numpy

RUN mkdir /tmp/work
WORKDIR /tmp/work
#RUN git clone --depth=1 https://github.com/opencv/opencv.git
#RUN git clone --depth=1 https://github.com/opencv/opencv_contrib.git

ENV OPENCV_VERSION="4.9.0"

RUN wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip &&\
    unzip $OPENCV_VERSION.zip &&\
    rm $OPENCV_VERSION.zip &&\
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip &&\
    unzip ${OPENCV_VERSION}.zip &&\
    rm ${OPENCV_VERSION}.zip

COPY sources.list /etc/apt/sources.list
RUN apt update -y
RUN dpkg --add-architecture arm64
RUN \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get -y install \
        libavcodec-dev:arm64 \
        libavformat-dev:arm64 \
        libavutil-dev:arm64 \
        libswscale-dev:arm64 \
        libfreetype-dev:arm64 \
        libharfbuzz-dev:arm64 \
        libpython3-dev:arm64 \
        libv4l-dev:arm64

COPY cc.sh /tmp/work/cc.sh
RUN chmod +x /tmp/work/cc.sh

ENTRYPOINT ["/bin/bash", "/tmp/work/cc.sh" ]