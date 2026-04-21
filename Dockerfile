# CUDA 11.6 driver (510.39.01) compatible image
# PyTorch 2.3.1+cu118 bundles its own CUDA runtime, so it works on driver >=450.80.02
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo

RUN apt-get update && apt-get install -y \
        wget curl git \
        ffmpeg libsm6 libxext6 libgl1-mesa-glx \
        build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Miniconda (Python 3.8)
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py38_23.3.1-0-Linux-x86_64.sh \
        -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && ${CONDA_DIR}/bin/conda clean -afy

ENV PATH=${CONDA_DIR}/bin:$PATH

WORKDIR /workspace

# ---- Install PyTorch first (pin CUDA 11.8 build that ships its own runtime) ----
RUN conda install -y \
        python=3.8.18 \
        pytorch==2.3.1 torchvision==0.18.1 pytorch-cuda=11.8 \
        -c pytorch -c nvidia \
    && conda clean -afy

# ---- Install remaining conda dependencies ----
RUN conda install -y -c conda-forge -c defaults \
        av=11.0.0 \
        decord=0.6.0 \
        einops=0.8.0 \
        imageio=2.34.2 \
        numpy=1.24.4 \
        pillow=11.1.0 \
        requests=2.32.3 \
        tqdm=4.66.4 \
    && conda clean -afy

# ---- Install pip-only packages ----
RUN pip install --no-cache-dir \
        peft==0.13.2 \
        pycocoevalcap==1.2 \
        sentence-transformers==3.0.1 \
        timm==1.0.12 \
        transformers==4.43.2 \
        wandb==0.19.1 \
        opencv-python==4.10.0.84 \
        SoccerNet==0.1.62 \
        huggingface_hub

# ---- Copy project ----
COPY . /workspace/

ENV PYTHONPATH=/workspace

CMD ["/bin/bash"]
