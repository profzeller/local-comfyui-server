"""
ComfyUI API Wrapper
Provides a simple REST API for common image generation tasks
"""

import asyncio
import base64
import json
import os
import uuid
from pathlib import Path
from typing import Optional

import aiofiles
import uvicorn
import websockets
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ComfyUI API", description="REST API wrapper for ComfyUI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_WS = "ws://127.0.0.1:8188/ws"
OUTPUT_DIR = Path("/app/ComfyUI/output")


class TextToImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = -1
    checkpoint: str = "sd_xl_base_1.0.safetensors"
    scheduler: str = "normal"
    sampler: str = "euler"


class ImageToImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    image_base64: str
    strength: float = 0.75
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = -1
    checkpoint: str = "sd_xl_base_1.0.safetensors"


class GenerationResponse(BaseModel):
    image_base64: str
    seed: int
    prompt: str
    width: int
    height: int


def get_text2img_workflow(params: TextToImageRequest) -> dict:
    """Generate a basic SDXL text-to-image workflow."""
    seed = params.seed if params.seed >= 0 else int.from_bytes(os.urandom(4), "big")

    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": params.cfg_scale,
                "denoise": 1,
                "latent_image": ["5", 0],
                "model": ["4", 0],
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": params.sampler,
                "scheduler": params.scheduler,
                "seed": seed,
                "steps": params.steps
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": params.checkpoint
            }
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "batch_size": 1,
                "height": params.height,
                "width": params.width
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": params.prompt
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": params.negative_prompt
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "api_output",
                "images": ["8", 0]
            }
        }
    }


async def queue_prompt(workflow: dict) -> str:
    """Queue a workflow and return the prompt ID."""
    import aiohttp

    client_id = str(uuid.uuid4())

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow, "client_id": client_id}
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise HTTPException(status_code=500, detail=f"Failed to queue prompt: {text}")
            data = await response.json()
            return data["prompt_id"], client_id


async def wait_for_completion(prompt_id: str, client_id: str, timeout: int = 300) -> dict:
    """Wait for workflow completion via websocket."""
    try:
        async with websockets.connect(f"{COMFYUI_WS}?clientId={client_id}") as ws:
            while True:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    data = json.loads(message)

                    if data.get("type") == "executing":
                        exec_data = data.get("data", {})
                        if exec_data.get("prompt_id") == prompt_id and exec_data.get("node") is None:
                            # Execution complete
                            return await get_history(prompt_id)

                    if data.get("type") == "execution_error":
                        raise HTTPException(status_code=500, detail=f"Execution error: {data}")

                except asyncio.TimeoutError:
                    raise HTTPException(status_code=504, detail="Generation timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WebSocket error: {str(e)}")


async def get_history(prompt_id: str) -> dict:
    """Get the history/output for a prompt."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{COMFYUI_URL}/history/{prompt_id}") as response:
            if response.status != 200:
                raise HTTPException(status_code=500, detail="Failed to get history")
            return await response.json()


async def get_output_image(history: dict, prompt_id: str) -> tuple[bytes, str]:
    """Extract the output image from history."""
    outputs = history.get(prompt_id, {}).get("outputs", {})

    for node_id, node_output in outputs.items():
        if "images" in node_output:
            image_info = node_output["images"][0]
            filename = image_info["filename"]
            subfolder = image_info.get("subfolder", "")

            # Read the image file
            if subfolder:
                image_path = OUTPUT_DIR / subfolder / filename
            else:
                image_path = OUTPUT_DIR / filename

            if image_path.exists():
                async with aiofiles.open(image_path, "rb") as f:
                    return await f.read(), filename

    raise HTTPException(status_code=500, detail="No output image found")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    import aiohttp

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{COMFYUI_URL}/system_stats", timeout=5) as response:
                if response.status == 200:
                    stats = await response.json()
                    return {
                        "status": "healthy",
                        "comfyui": "connected",
                        "vram_used": stats.get("devices", [{}])[0].get("vram_used_gb", 0),
                        "vram_total": stats.get("devices", [{}])[0].get("vram_total_gb", 0),
                    }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

    return {"status": "unhealthy", "error": "ComfyUI not responding"}


@app.get("/models")
async def list_models():
    """List available checkpoint models."""
    models_dir = Path("/app/ComfyUI/models/checkpoints")
    models = []

    if models_dir.exists():
        for f in models_dir.glob("**/*"):
            if f.suffix.lower() in [".safetensors", ".ckpt", ".pt"]:
                models.append(f.name)

    return {"models": models}


@app.post("/generate", response_model=GenerationResponse)
async def generate_image(request: TextToImageRequest):
    """Generate an image from text prompt."""

    # Build workflow
    workflow = get_text2img_workflow(request)

    # Queue the prompt
    prompt_id, client_id = await queue_prompt(workflow)

    # Wait for completion
    history = await wait_for_completion(prompt_id, client_id)

    # Get output image
    image_bytes, filename = await get_output_image(history, prompt_id)

    # Extract seed from workflow
    seed = workflow["3"]["inputs"]["seed"]

    return GenerationResponse(
        image_base64=base64.b64encode(image_bytes).decode("utf-8"),
        seed=seed,
        prompt=request.prompt,
        width=request.width,
        height=request.height,
    )


@app.post("/workflow")
async def run_custom_workflow(workflow: dict):
    """Run a custom ComfyUI workflow."""

    # Queue the prompt
    prompt_id, client_id = await queue_prompt(workflow)

    # Wait for completion
    history = await wait_for_completion(prompt_id, client_id)

    # Try to get output image
    try:
        image_bytes, filename = await get_output_image(history, prompt_id)
        return {
            "status": "completed",
            "prompt_id": prompt_id,
            "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
        }
    except HTTPException:
        # No image output, return history
        return {
            "status": "completed",
            "prompt_id": prompt_id,
            "history": history.get(prompt_id, {}),
        }


if __name__ == "__main__":
    # Install aiohttp if not present
    try:
        import aiohttp
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "aiohttp"])

    uvicorn.run(app, host="0.0.0.0", port=8189)
