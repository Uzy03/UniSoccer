FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo

RUN apt-get update && apt-get install -y \
        wget curl git \
        ffmpeg libsm6 libxext6 libgl1-mesa-glx \
        build-essential ca-certificates \
        pkg-config \
        libavformat-dev libavcodec-dev libavdevice-dev \
        libavutil-dev libswscale-dev libswresample-dev libavfilter-dev \
        python3 python3-pip \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---- PyTorch (match host driver CUDA compatibility) ----
RUN pip install --no-cache-dir \
        typing-extensions==4.14.1 \
        torch==1.13.1+cu116 \
        torchvision==0.14.1+cu116 \
        --index-url https://download.pytorch.org/whl/cu116 \
        --extra-index-url https://pypi.org/simple

# ---- All other dependencies ----
RUN pip install --no-cache-dir \
        av==11.0.0 \
        decord==0.6.0 \
        einops==0.8.0 \
        imageio==2.34.2 \
        numpy==1.24.4 \
        pillow==10.4.0 \
        requests==2.32.3 \
        tqdm==4.66.4 \
        typing-extensions==4.14.1 \
        peft==0.13.2 \
        pycocoevalcap==1.2 \
        sentence-transformers==3.0.1 \
        timm==1.0.12 \
        transformers==4.43.2 \
        wandb==0.19.1 \
        opencv-python==4.10.0.84 \
        SoccerNet==0.1.62 \
        huggingface_hub

COPY . /workspace/

ENV PYTHONPATH=/workspace

CMD ["/bin/bash"]
