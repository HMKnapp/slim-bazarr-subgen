FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y python3-pip ffmpeg build-essential libcudnn8

RUN apt-get install -y vim nvtop
RUN apt-get clean

COPY requirements.txt /requirements.txt
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV NVIDIA_DISABLE_REQUIRE=1
ENTRYPOINT [ "/entrypoint.sh" ]
