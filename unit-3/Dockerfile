FROM nvcr.io/nvidia/pytorch:22.11-py3

# Configure timezone for apt-get install (avoids the interactive promp during installation)
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
# Install system dependencies
RUN apt-get update && apt-get install -y \
    swig \
    cmake \
    python-opengl \
    ffmpeg \
    xvfb \
    freeglut3-dev \
    git-lfs

# Install python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /workspace

# Clone the repository and install the python dependencies
RUN git clone https://github.com/DLR-RM/rl-baselines3-zoo \
    && cd rl-baselines3-zoo \
    && pip install -r requirements.txt
