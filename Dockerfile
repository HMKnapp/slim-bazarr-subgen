FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y python3-pip libcudnn8
RUN apt-get clean

RUN apt remove -y --allow-remove-essential cuda-compat-12-4 cuda-cudart-12-4 cuda-cudart-dev-12-4 cuda-keyring cuda-libraries-12-4 cuda-libraries-dev-12-4 cuda-nsight-compute-12-4 cuda-nvml-dev-12-4 cuda-nvprof-12-4 cuda-nvtx-12-4 ncurses-base ncurses-bin e2fsprogs
RUN apt autoremove -y

COPY requirements.txt /requirements.txt
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT [ "/entrypoint.sh" ]
