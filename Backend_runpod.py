"""
RunPod Serverless Handler - Botanical Illustration Generator
SDXL + ControlNet + KappaNeuro Botanical LoRA
"""

import runpod
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from controlnet_aux import CannyDetector
from PIL import Image
import io
import base64

# Global pipeline (loaded once, reused for all requests)
pipe = None
canny_detector = None
blip_processor = None
blip_model = None

def load_models():
    """Load models once at startup"""
    global pipe, canny_detector, blip_processor, blip_model
    
    print("Loading ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16
    )
    
    print("Loading SDXL...")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    
    print("Loading Botanical LoRA...")
    pipe.load_lora_weights("KappaNeuro/century-botanical-illustration")
    
    print("Loading Canny detector...")
    canny_detector = CannyDetector()
    
    print("Loading BLIP-2 for descriptions...")
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16
    ).to("cuda")
    
    print("✓ All models loaded!")

def generate_botanical(sketch_base64):
    """
    Generate botanical illustration from sketch
    
    Args:
        sketch_base64: Base64 encoded sketch image
        
    Returns:
        Dict with base64 image, plant name, and description
    """
    global pipe, canny_detector, blip_processor, blip_model
    
    # Decode sketch
    sketch_data = base64.b64decode(sketch_base64)
    sketch = Image.open(io.BytesIO(sketch_data)).convert("RGB")
    sketch = sketch.resize((1024, 1024))
    
    # Process with Canny
    canny_image = canny_detector(sketch)
    
    # Fixed prompts (user doesn't control these)
    prompt = "single botanical illustration of one plant, vintage scientific diagram, clean white background, detailed watercolor"
    negative_prompt = "multiple plants, many flowers, bouquet, arrangement, group of plants, cluster, photo, realistic, modern, dark background, messy, text, border"
    
    # Generate botanical illustration
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=canny_image,
        num_inference_steps=30,
        controlnet_conditioning_scale=0.5,
        guidance_scale=7.5
    ).images[0]
    
    # Generate botanical description using BLIP-2
    print("Generating botanical description...")
    inputs_desc = blip_processor(image, text="Describe this botanical illustration in scientific detail:", return_tensors="pt").to("cuda", torch.float16)
    generated_ids_desc = blip_model.generate(**inputs_desc, max_new_tokens=100)
    description = blip_processor.decode(generated_ids_desc[0], skip_special_tokens=True)
    
    # Generate Latin plant name using BLIP-2
    print("Generating plant name...")
    inputs_name = blip_processor(image, text="What is the Latin botanical name for this plant?", return_tensors="pt").to("cuda", torch.float16)
    generated_ids_name = blip_model.generate(**inputs_name, max_new_tokens=20)
    plant_name = blip_processor.decode(generated_ids_name[0], skip_special_tokens=True).strip()
    
    # Fallback if BLIP-2 doesn't generate proper Latin name
    if len(plant_name.split()) > 3 or not plant_name[0].isupper():
        import random
        genus_options = ["Rosa", "Florensis", "Botanica", "Herbalis", "Plantae"]
        species_options = ["elegans", "magnifica", "pristina", "botanicus", "illustris"]
        plant_name = f"{random.choice(genus_options)} {random.choice(species_options)}"
    
    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return {
        "image": img_base64,
        "plant_name": plant_name,
        "description": description
    }

def handler(event):
    """
    RunPod handler function
    
    Input format:
    {
        "input": {
            "sketch": "base64_encoded_image_string"
        }
    }
    
    Output format:
    {
        "image": "base64_encoded_generated_image",
        "plant_name": "Rosa elegans",
        "description": "Scientific description of the botanical specimen..."
    }
    """
    try:
        # Get sketch from input
        sketch_base64 = event["input"]["sketch"]
        
        # Generate botanical illustration + description
        result = generate_botanical(sketch_base64)
        
        return {
            "image": result["image"],
            "plant_name": result["plant_name"],
            "description": result["description"],
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

# Load models on cold start
load_models()

# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
