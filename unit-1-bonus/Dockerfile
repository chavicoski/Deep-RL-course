# Using pytorch 1.8 to comply with the ml-agents dependencies
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

# Update the apt-key (the base image it's old) and download some system dependencies
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-get update && apt-get install -y git wget unzip

# Place the code in the target workspace
WORKDIR /workspace

# Clone the repository and install the python dependencies
RUN git clone https://github.com/huggingface/ml-agents \
    && cd ml-agents \
    && pip install protobuf==3.9.2 \
    && pip install -e ./ml-agents-envs \
    && pip install -e ./ml-agents

# Download the environment
RUN cd ml-agents \
    && mkdir -p mkdir trained-envs-executables/linux \
    && wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zv3M95ZJTWHUVOWT6ckq_cm98nft8gdF' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zv3M95ZJTWHUVOWT6ckq_cm98nft8gdF" -O ./trained-envs-executables/linux/Huggy.zip \
    && rm -rf /tmp/cookies.txt \
    && unzip -d ./trained-envs-executables/linux/ ./trained-envs-executables/linux/Huggy.zip \
    && chmod -R 755 ./trained-envs-executables/linux/Huggy
