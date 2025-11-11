"""
RunPod Serverless Handler - Botanical Illustration Generator
Based on your working Colab notebook
"""

import runpod
import torch
import io
import base64
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from controlnet_aux import CannyDetector

# Global variables for models (loaded once on cold start)
pipe = None
canny_detector = None

def load_models():
    """Load all models once at startup"""
    global pipe, canny_detector
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16
    )
    
    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    
    print("Loading botanical LoRA...")
    pipe.load_lora_weights("KappaNeuro/century-botanical-illustration")
    
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
