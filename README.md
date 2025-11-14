# Botanical Illustration Generator

A serverless AI interface that transforms hand-drawn sketches into detailed botanical illustrations using Stable Diffusion XL, ControlNet, and a specialized botanical LoRA. Deployed on RunPod serverless.

## 🌿 Features

- **Sketch-to-Illustration**: Convert simple line drawings into detailed botanical illustrations
- **ControlNet Integration**: Uses Canny edge detection to preserve sketch structure
- **Botanical LoRA**: Specialized model trained on botanical illustrations
- **Serverless Architecture**: Deployed on RunPod for automatic scaling
- **Network Volume Caching**: Persistent model storage for fast cold starts

## 🚀 RunPod Deployment

### Prerequisites

- RunPod account
- Docker (for building images)

### Step 1: Build Docker Image

```bash
docker build -t botanical-illustration:latest .
```

### Step 2: Push to Container Registry

Push your image to Docker Hub, GitHub Container Registry, or another registry accessible to RunPod.

```bash
docker tag botanical-illustration:latest yourusername/botanical-illustration:latest
docker push yourusername/botanical-illustration:latest
```

### Step 3: Create Network Volume (Optional)

**Note on Network Volumes:**
The code attempts to use network volumes to reduce cold start times by caching models persistently. However, **a reliable solution has not been found** due to limitations in how the SDXL pipeline and LoRA adapters handle model saving/loading.

**Why This Matters:**
- Models are large (~10-12GB)
- Without proper caching, they download on every cold start
- The SDXL pipeline and LoRA adapter require modifications to their internal layers to be saved to a network volume
- Current implementation uses HuggingFace's automatic caching, but this doesn't reliably persist across all scenarios

**What This Means:**
- **Models will still work** - the application functions correctly
- **First image generation takes ~3 minutes** - models download on the first request
- Network volumes may help in some cases, but results are inconsistent

**If You Want to Try Network Volumes:**

1. Go to **RunPod Dashboard** → **Storage** → **Network Volumes**
2. Click **"Create Network Volume"**
3. Configure:
   - **Name**: `botanical-models-cache` (or your preferred name)
   - **Size**: **25GB minimum** (50GB recommended)
   - **Region**: Choose the same region as your endpoint
4. Click **"Create"**

### Step 4: Create Serverless Endpoint

1. Go to **RunPod Dashboard** → **Serverless** → **Endpoints**
2. Click **"New Endpoint"**
3. Configure:
   - **Name**: Your endpoint name
   - **Container Image**: `yourusername/botanical-illustration:latest`
   - **GPU Type**: Select L4, RTX A5000, or RTX 3090
4. Scroll to **"Advanced"** section
5. (Optional) Under **"Network Volume"**, select your network volume
   - Mount path will automatically be `/runpod-volume`
6. Click **"Save Endpoint"**

### Step 5: Expected Behavior

**First Request:**
- Models download from HuggingFace Hub
- Takes approximately **3 minutes** for the first image generation
- Models may be cached by HuggingFace, but this is not guaranteed to persist

**Subsequent Requests:**
- If models are cached, subsequent requests are faster
- If cache is lost, models re-download (another ~3 minutes)
- Network volume may help, but results vary

## ⚠️ Cold Start & Caching Limitations

### The Problem

Despite my attempts to use network volumes for persistent model caching, **a reliable solution has not been found**. The core issue is that:

1. **SDXL Pipeline**: The SDXL pipeline with ControlNet integration requires modifications to its internal layers to be properly saved to a network volume. The standard `save_pretrained()` method doesn't work reliably for this use case.

2. **LoRA Adapter**: Similarly, the LoRA adapter needs changes to its layer structure to be saved and loaded from a network volume. Format compatibility issues prevent reliable caching.

### Current Implementation

The handler uses HuggingFace's automatic caching system with `cache_dir` pointing to the network volume. However, this doesn't guarantee persistence across all scenarios. ( I have not been able to resolve this for myself)

### What This Means for You

- ✅ **The application works correctly** - all functionality is intact
- ⏱️ **First image takes ~3 minutes** - models download on the first request
- 🔄 **Subsequent requests may be faster** - if HuggingFace cache persists
- ❌ **No guaranteed cold start optimization** - models may re-download on each cold start

### Contributing Solutions

If you find a reliable workaround for:
- Saving/loading SDXL+ControlNet pipelines to network volumes
- Caching LoRA adapters persistently
- Modifying model layers for network volume compatibility

Please open an issue or submit a PR! This would significantly improve cold start times.

## 📡 API Usage

### Input Format

```json
{
  "input": {
    "sketch": "base64_encoded_sketch_image_string"
  }
}
```

### Output Format

```json
{
  "image": "base64_encoded_botanical_illustration",
  "status": "success"
}
```

### Example Request

```bash
curl -X POST \
  https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "input": {
      "sketch": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  }'
```


## 📋 Requirements

See `requirements.txt` for Python dependencies. Key packages:
- `diffusers==0.25.0`
- `transformers==4.35.2`
- `controlnet-aux==0.0.7`
- `runpod`

## 🏗️ Architecture

- **Base Model**: Stable Diffusion XL (SDXL) 1.0
- **ControlNet**: Canny edge detection for structure preservation
- **LoRA**: Century Botanical Illustration by KappaNeuro
- **Platform**: RunPod Serverless
- **Framework**: PyTorch, Diffusers

## 📝 License

MIT License

## 🙏 Acknowledgments

- **Stable Diffusion XL**: Stability AI
- **ControlNet**: lllyasviel
- **Botanical LoRA**: KappaNeuro/century-botanical-illustration
- **RunPod**: Serverless GPU platform
