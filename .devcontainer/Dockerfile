
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

ENV TZ=Europe/Stockholm
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update -y && apt-get install -y \
    && apt-get install build-essential -y \
    && apt-get install gdb -y \
    && apt-get install git-gui -y\
    && groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support for the non-root user
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt-get install python-pip -y \
    && pip install imageio\
    && pip install future
RUN pip install conda

RUN pip install torch torchvision 

RUN pip install --upgrade pip && pip install pip install opencv-contrib-python
RUN pip install -U matplotlib
RUN pip install -U scikit-image
RUN pip install numpy==1.13.1 scipy matplotlib ipython jupyter pandas sympy nose
RUN pip install autopep8 
RUN pip install pydensecrf
RUN pip install pyquaternion==0.9.0
RUN apt-get install python-tk -y

VOLUME data
VOLUME database
VOLUME project

ENV DEBIAN_FRONTEND=dialog





