FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo

RUN apt-get update && apt-get install -y \
        wget curl git \
        ffmpeg libsm6 libxext6 libgl1-mesa-glx \
        build-essential ca-certificates \
        python3 python3-pip \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---- PyTorch (CUDA 11.8 build bundles its own runtime) ----
RUN pip install --no-cache-dir \
        torch==2.3.1+cu118 \
        torchvision==0.18.1+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

# ---- All other dependencies ----
RUN pip install --no-cache-dir \
        av==11.0.0 \
        decord==0.6.0 \
        einops==0.8.0 \
        imageio==2.34.2 \
        numpy==1.24.4 \
        pillow==11.1.0 \
        requests==2.32.3 \
        tqdm==4.66.4 \
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
