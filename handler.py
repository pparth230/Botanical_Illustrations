"""
RunPod Serverless Handler - Botanical Illustration Generator
Based on your working Colab notebook
"""

import runpod
import torch
import io
import base64
import os
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from controlnet_aux import CannyDetector

# Network volume path (RunPod mounts network volumes here)
# Network volumes are PERSISTENT - models will remain across:
# - Container restarts
# - Container destruction/recreation  
# - Endpoint updates
# - Pod shutdowns
# They only disappear if you manually delete the volume
NETWORK_VOLUME_PATH = os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume")
MODELS_CACHE_DIR = os.path.join(NETWORK_VOLUME_PATH, "models_cache")

# Global variables for models (loaded once on cold start)
pipe = None
canny_detector = None

def ensure_cache_dir():
    """Create cache directory if it doesn't exist"""
    Path(MODELS_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Using cache directory: {MODELS_CACHE_DIR}")
    # Verify network volume is accessible
    if os.path.exists(NETWORK_VOLUME_PATH):
        print(f"✓ Network volume found at: {NETWORK_VOLUME_PATH}")
    else:
        print(f"⚠ Network volume not found at {NETWORK_VOLUME_PATH}, will use container storage (not persistent)")

def load_models():
    """Load all models once at startup, using network volume cache if available"""
    global pipe, canny_detector
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Ensure cache directory exists
    ensure_cache_dir()
    
    # Set HuggingFace cache to use network volume directly
    # This ensures all downloads go straight to network volume, not container disk
    os.environ["HF_HOME"] = MODELS_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(MODELS_CACHE_DIR, "transformers")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(MODELS_CACHE_DIR, "datasets")
    # Set temp directory to network volume to avoid filling container disk
    temp_dir = os.path.join(MODELS_CACHE_DIR, "tmp")
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = temp_dir
    os.environ["TMP"] = temp_dir
    os.environ["TEMP"] = temp_dir
    
    # Set cache directories for models
    controlnet_cache = os.path.join(MODELS_CACHE_DIR, "controlnet-canny-sdxl")
    sdxl_cache = os.path.join(MODELS_CACHE_DIR, "stable-diffusion-xl-base-1.0")
    lora_cache = os.path.join(MODELS_CACHE_DIR, "century-botanical-illustration")
    
    print("Loading ControlNet...")
    # Check if cached (verify config.json exists to ensure it's a valid model)
    if os.path.exists(controlnet_cache) and os.path.exists(os.path.join(controlnet_cache, "config.json")):
        print(f"  ✓ Loading from cache: {controlnet_cache}")
        controlnet = ControlNetModel.from_pretrained(
            controlnet_cache,
            torch_dtype=torch.float16,
            local_files_only=True
        )
    else:
        print("  ⚠ Downloading ControlNet (this may take a while)...")
        # Use cache_dir to download directly to network volume
        # HuggingFace will cache automatically, no need for extra save_pretrained
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16,
            cache_dir=MODELS_CACHE_DIR
        )
        # Try to save to organized location (skip if it fails due to space)
        try:
            controlnet.save_pretrained(controlnet_cache)
            print(f"  ✓ Cached to: {controlnet_cache}")
        except OSError as e:
            print(f"  Note: Could not save to organized cache (models still cached by HuggingFace): {str(e)}")
    
    print("Loading SDXL pipeline...")
    # Check if cached (verify model_index.json exists to ensure it's a valid model)
    if os.path.exists(sdxl_cache) and os.path.exists(os.path.join(sdxl_cache, "model_index.json")):
        print(f"  ✓ Loading from cache: {sdxl_cache}")
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            sdxl_cache,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
            local_files_only=True
        ).to(device)
    else:
        print("  ⚠ Downloading SDXL pipeline (this may take a while)...")
        # Use cache_dir to download directly to network volume
        # HuggingFace will cache automatically, no need for extra save_pretrained
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
            cache_dir=MODELS_CACHE_DIR
        ).to(device)
        # Try to save to organized location (skip if it fails due to space)
        try:
            pipe.save_pretrained(sdxl_cache)
            print(f"  ✓ Cached to: {sdxl_cache}")
        except OSError as e:
            print(f"  Note: Could not save to organized cache (models still cached by HuggingFace): {str(e)}")
    
    print("Loading botanical LoRA...")
    # LoRA weights - try to load from cache, fallback to download
    # Note: LoRA caching behavior varies by diffusers version
    try:
        if os.path.exists(lora_cache):
            print(f"  Attempting to load LoRA from cache: {lora_cache}")
            pipe.load_lora_weights(lora_cache, local_files_only=True)
            print(f"  ✓ LoRA loaded from cache")
        else:
            raise FileNotFoundError("LoRA cache not found")
    except (FileNotFoundError, Exception) as e:
        print(f"  ⚠ LoRA not in cache or cache load failed, downloading...")
        print(f"  Error: {str(e)}")
        pipe.load_lora_weights("KappaNeuro/century-botanical-illustration")
        # Try to save LoRA weights if possible
        try:
            # Some versions support saving LoRA weights
            if hasattr(pipe, 'save_lora_weights'):
                Path(lora_cache).mkdir(parents=True, exist_ok=True)
                pipe.save_lora_weights(lora_cache)
                print(f"  ✓ LoRA cached to: {lora_cache}")
            else:
                print(f"  Note: LoRA caching not supported in this diffusers version")
        except Exception as save_error:
            print(f"  Note: Could not cache LoRA: {str(save_error)}")
        print(f"  ✓ LoRA loaded")
    
    print("Loading Canny detector...")
    canny_detector = CannyDetector()
    
    print("✓ All models loaded successfully!")

def generate_botanical(sketch_base64):
    """
    Generate botanical illustration from sketch
    
    Args:
        sketch_base64: Base64 encoded sketch image string
        
    Returns:
        Base64 encoded botanical illustration
    """
    global pipe, canny_detector
    
    # Decode the sketch image
    sketch_data = base64.b64decode(sketch_base64)
    sketch = Image.open(io.BytesIO(sketch_data)).convert("RGB")
    sketch = sketch.resize((1024, 1024))
    
    # Process sketch with Canny edge detection
    print("Processing sketch with Canny detector...")
    canny_image = canny_detector(sketch)
    
    # Generate botanical illustration
    print("Generating botanical illustration...")
    prompt = "botanical illustration of a plant, vintage scientific diagram, clean white background, detailed watercolor"
    negative_prompt = "multiple plants, many flowers, bouquet, arrangement, group of plants, cluster, photo, realistic, modern, dark background, messy, text, border"
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=canny_image,
        num_inference_steps=30,
        controlnet_conditioning_scale=0.5,
        guidance_scale=7.5
    ).images[0]
    
    # Convert result to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    print("✓ Generation complete!")
    return img_base64

def handler(event):
    """
    RunPod handler function
    
    Expected input format:
    {
        "input": {
            "sketch": "base64_encoded_sketch_string"
        }
    }
    
    Output format:
    {
        "image": "base64_encoded_botanical_illustration",
        "status": "success"
    }
    """
    try:
        # Extract sketch from input
        sketch_base64 = event["input"]["sketch"]
        
        # Generate botanical illustration
        result_image = generate_botanical(sketch_base64)
        
        return {
            "image": result_image,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }

# Load models on container startup (cold start)
print("Starting model initialization...")
load_models()
print("Models ready. Starting RunPod handler...")

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})
