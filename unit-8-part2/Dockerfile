FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

# Configure timezone for apt-get install (avoids the interactive promp during installation)
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
    nasm tar libbz2-dev libgtk-3-dev cmake git libfluidsynth-dev libgme-dev \
    libopenal-dev timidity libwildmidi-dev unzip libboost-all-dev \
    g++ make libmpg123-dev libsndfile1-dev chrpath python3-dev python3-pip \
    ffmpeg xvfb

# Install python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -U pip
RUN pip install -r /tmp/requirements.txt

WORKDIR /workspace
