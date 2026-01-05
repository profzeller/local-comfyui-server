# ComfyUI Local Server
# Image generation with ComfyUI on local GPU

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

WORKDIR /app/ComfyUI

# Install ComfyUI requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install additional useful packages
RUN pip install --no-cache-dir \
    opencv-python \
    insightface \
    onnxruntime-gpu \
    segment-anything \
    groundingdino-py \
    fastapi \
    uvicorn \
    python-multipart \
    aiofiles \
    websockets

# Install ComfyUI Manager for easy custom node installation
RUN cd custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# Create directories for models (will be mounted as volumes)
RUN mkdir -p models/checkpoints models/vae models/loras models/controlnet \
    models/clip models/clip_vision models/upscale_models models/embeddings

# Copy API wrapper
COPY api_wrapper.py /app/api_wrapper.py

# Expose ports
EXPOSE 8188 8189

# Start ComfyUI and API wrapper
CMD python main.py --listen 0.0.0.0 --port 8188 & \
    sleep 10 && \
    python /app/api_wrapper.py
