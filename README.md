# Local ComfyUI Server

> Part of the [P16 GPU Server](https://github.com/profzeller/p16-server-setup) ecosystem

Dedicated image generation server using ComfyUI on a local GPU.

Optimized for 16GB VRAM GPUs to run SDXL and other image generation models.

## Features

- Full ComfyUI with web UI
- REST API wrapper for easy integration
- Support for SDXL (recommended for 16GB)
- LoRA, ControlNet, and custom node support
- ComfyUI Manager for easy extension installation

## Requirements

- NVIDIA GPU with 16GB VRAM
- Ubuntu Server 22.04+ (or any Linux with Docker)
- Docker & Docker Compose
- NVIDIA Driver 525+
- NVIDIA Container Toolkit
- 50-200GB disk space for models

## VRAM Usage

| Model | VRAM | Fits 16GB? |
|-------|------|------------|
| SDXL 1.0 | ~6-8 GB | Yes (recommended) |
| SDXL + LoRA | ~8-10 GB | Yes |
| SDXL + ControlNet | ~10-12 GB | Yes |
| Flux Schnell (fp8) | ~12 GB | Tight fit |
| Flux Dev (fp8) | ~12 GB | Tight fit |
| Flux Dev (fp16) | ~24 GB | No |

**Recommendation for 16GB:** Use SDXL with LoRAs for best results.

## Quick Start

### 1. Install NVIDIA Container Toolkit

```bash
# Add NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. Clone and Setup

```bash
git clone https://github.com/profzeller/local-comfyui-server.git
cd local-comfyui-server

# Create model directories
mkdir -p models/checkpoints models/vae models/loras models/controlnet
mkdir -p models/clip models/clip_vision models/upscale_models models/embeddings
mkdir -p custom_nodes output input
```

### 3. Download Models

**SDXL (Recommended starter):**

```bash
# SDXL Base
wget -P models/checkpoints \
  https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

# SDXL Refiner (optional)
wget -P models/checkpoints \
  https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors

# SDXL VAE
wget -P models/vae \
  https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors
```

**Flux Dev (requires HuggingFace login):**

```bash
# Login to HuggingFace first
huggingface-cli login

# Flux Dev (fp8 for 24GB GPUs)
huggingface-cli download black-forest-labs/FLUX.1-dev \
  --local-dir models/checkpoints/flux-dev \
  --include "flux1-dev.safetensors"
```

### 4. Start the Server

```bash
docker compose up -d
```

First start will build the image (~10-15 min).

## Accessing the Server

| Service | URL | Description |
|---------|-----|-------------|
| ComfyUI Web UI | http://localhost:8188 | Full ComfyUI interface |
| REST API | http://localhost:8189 | Simplified REST API |

From other machines on your network:
```
http://<server-ip>:8188  # Web UI
http://<server-ip>:8189  # API
```

## REST API Usage

### Health Check

```bash
curl http://localhost:8189/health
```

### List Models

```bash
curl http://localhost:8189/models
```

### Generate Image (Text-to-Image)

```bash
curl -X POST http://localhost:8189/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene mountain landscape at sunrise, wellness retreat, peaceful",
    "negative_prompt": "ugly, blurry, low quality",
    "width": 1024,
    "height": 1024,
    "steps": 20,
    "cfg_scale": 7.0,
    "checkpoint": "sd_xl_base_1.0.safetensors"
  }'
```

### Save Generated Image

```bash
curl -s -X POST http://localhost:8189/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "wellness meditation scene"}' \
  | jq -r '.image_base64' | base64 -d > output.png
```

### Run Custom Workflow

```bash
curl -X POST http://localhost:8189/workflow \
  -H "Content-Type: application/json" \
  -d @my_workflow.json
```

### Python Example

```python
import requests
import base64

response = requests.post("http://localhost:8189/generate", json={
    "prompt": "A peaceful yoga studio with natural light, wellness aesthetic",
    "negative_prompt": "dark, cluttered, ugly",
    "width": 1024,
    "height": 1024,
    "steps": 25,
    "cfg_scale": 7.5,
})

data = response.json()

# Save the image
with open("output.png", "wb") as f:
    f.write(base64.b64decode(data["image_base64"]))

print(f"Generated image with seed: {data['seed']}")
```

## API Reference

### POST /generate

Generate an image from a text prompt.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description of the image |
| `negative_prompt` | string | "" | What to avoid in the image |
| `width` | int | 1024 | Image width |
| `height` | int | 1024 | Image height |
| `steps` | int | 20 | Inference steps |
| `cfg_scale` | float | 7.0 | Classifier-free guidance scale |
| `seed` | int | -1 | Random seed (-1 for random) |
| `checkpoint` | string | "sd_xl_base_1.0.safetensors" | Model checkpoint |
| `sampler` | string | "euler" | Sampling method |
| `scheduler` | string | "normal" | Scheduler type |

### POST /workflow

Run a custom ComfyUI workflow (JSON format).

### GET /models

List available checkpoint models.

### GET /health

Check server status and VRAM usage.

## Directory Structure

```
local-comfyui-server/
├── docker-compose.yml
├── Dockerfile
├── api_wrapper.py
├── models/
│   ├── checkpoints/     # Main model files (.safetensors)
│   ├── vae/             # VAE models
│   ├── loras/           # LoRA models
│   ├── controlnet/      # ControlNet models
│   ├── clip/            # CLIP models
│   ├── clip_vision/     # CLIP vision models
│   ├── upscale_models/  # Upscaling models
│   └── embeddings/      # Textual embeddings
├── custom_nodes/        # ComfyUI extensions
├── output/              # Generated images
└── input/               # Input images for img2img
```

## Installing Custom Nodes

### Via ComfyUI Manager (Recommended)

1. Open ComfyUI web UI: http://localhost:8188
2. Click "Manager" button
3. Browse and install nodes

### Manually

```bash
cd custom_nodes
git clone https://github.com/user/ComfyUI-SomeExtension.git
docker compose restart
```

## Management

```bash
# View logs
docker compose logs -f

# Restart
docker compose restart

# Stop
docker compose down

# Rebuild after changes
docker compose build --no-cache
docker compose up -d

# Update ComfyUI
docker compose down
docker compose build --no-cache
docker compose up -d
```

## Troubleshooting

### Out of VRAM

- Use smaller models or fp8 quantization
- Reduce image dimensions
- Check VRAM usage: `nvidia-smi`
- Restart to clear VRAM: `docker compose restart`

### Model not found

- Ensure model is in the correct directory under `models/`
- Check the model filename matches what you're requesting
- List available models: `curl http://localhost:8189/models`

### Slow generation

- First generation loads the model (~30-60s)
- Subsequent generations are faster
- Use fewer steps for preview quality

### WebSocket errors

- Ensure ports 8188 and 8189 are not blocked
- Check firewall: `sudo ufw allow 8188 && sudo ufw allow 8189`

## License

MIT License - Use freely for personal and commercial projects.

## Credits

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - The most powerful Stable Diffusion GUI
- [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) - Custom node manager
