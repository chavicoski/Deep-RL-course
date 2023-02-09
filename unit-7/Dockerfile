# Using pytorch 1.8 to comply with the ml-agents dependencies
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

# Update the apt-key (the base image it's old) and download some system dependencies
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-get update && apt-get install -y git wget unzip git-lfs

# Place the code in the target workspace
WORKDIR /workspace

# Clone the repository and install the python dependencies
RUN git clone https://github.com/huggingface/ml-agents \
    && cd ml-agents \
    && pip install protobuf==3.9.2 \
    && pip install -e ./ml-agents-envs \
    && pip install -e ./ml-agents

# Download the environment SoccerTwos
RUN cd ml-agents \
    && mkdir -p mkdir training-envs-executables/linux \
    && wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1KuqBKYiXiIcU4kNMqEzhgypuFP5_45CL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1KuqBKYiXiIcU4kNMqEzhgypuFP5_45CL" -O ./training-envs-executables/linux/SoccerTwos.zip \
    && unzip -d ./training-envs-executables/linux/SoccerTwos ./training-envs-executables/linux/SoccerTwos.zip \
    && chmod -R 755 ./training-envs-executables/linux/SoccerTwos \
    && rm -rf /tmp/cookies.txt
