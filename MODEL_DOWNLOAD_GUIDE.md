# 📥 MiniMax-Remover Model Download Guide

## Overview
This guide explains how to download and set up the required models for the MiniMax-Remover node in ComfyUI.

## 🎯 Required Models

The MiniMax-Remover requires three types of models:

1. **VAE (Video Autoencoder)** - Encodes/decodes video frames
2. **Transformer** - The main diffusion model for object removal  
3. **Scheduler** - Controls the denoising process

## 🔗 Official Model Repository

**Official Hugging Face Repository**: [`zibojia/minimax-remover`](https://huggingface.co/zibojia/minimax-remover)

## 🚀 Quick Download Methods

### Method 1: Automatic Download Script (Recommended)

The project includes an automatic download script:

```bash
# Navigate to your MiniMax-Remover directory
cd MiniMax-Remover

# Run the download script
python download_models.py
```

This will automatically download all required models to the correct directories.

### Method 2: Hugging Face CLI (Manual)

```bash
# Install huggingface-cli if not already installed
pip install huggingface_hub

# Download all models at once
huggingface-cli download zibojia/minimax-remover --local-dir ./models --local-dir-use-symlinks False

# Or download specific components
huggingface-cli download zibojia/minimax-remover --include "vae/*" --local-dir ./models
huggingface-cli download zibojia/minimax-remover --include "transformer/*" --local-dir ./models  
huggingface-cli download zibojia/minimax-remover --include "scheduler/*" --local-dir ./models
```

### Method 3: Git Clone with LFS

```bash
# Install git-lfs if not already installed
git lfs install

# Clone the repository
git clone https://huggingface.co/zibojia/minimax-remover models
```

## 📁 Directory Structure

After downloading, your directory structure should look like:

```
MiniMax-Remover/
├── models/
│   ├── vae/
│   │   ├── config.json
│   │   ├── diffusion_pytorch_model.safetensors
│   │   └── ...
│   ├── transformer/
│   │   ├── config.json
│   │   ├── diffusion_pytorch_model.safetensors
│   │   └── ...
│   └── scheduler/
│       ├── scheduler_config.json
│       └── ...
└── ... (other project files)
```

## ⚙️ ComfyUI Integration

### Default Paths

By default, the node looks for models in these paths:

- **VAE**: `models/vae/`
- **Transformer**: `models/transformer/`
- **Scheduler**: `models/scheduler/`

### Alternative Paths

You can specify custom paths in the node interface:

1. **ComfyUI Models Directory**: Place models in `ComfyUI/models/`:
   ```
   ComfyUI/
   └── models/
       ├── vae/
       ├── transformer/
       └── scheduler/
   ```

2. **Absolute Paths**: Use full paths to model directories:
   ```
   VAE Path: /path/to/your/models/vae
   Transformer Path: /path/to/your/models/transformer
   Scheduler Path: /path/to/your/models/scheduler
   ```

## 🔧 Node Configuration

### Input Parameters

When using the node, configure these paths:

- **vae_path**: Path to VAE model directory
- **transformer_path**: Path to Transformer model directory  
- **scheduler_path**: Path to Scheduler config directory

### Example Configuration

```python
# Default configuration (models in project directory)
vae_path = "models/vae/"
transformer_path = "models/transformer/"
scheduler_path = "models/scheduler/"

# ComfyUI models directory
vae_path = "ComfyUI/models/vae/"
transformer_path = "ComfyUI/models/transformer/"
scheduler_path = "ComfyUI/models/scheduler/"

# Custom absolute paths
vae_path = "/my/custom/path/vae/"
transformer_path = "/my/custom/path/transformer/"
scheduler_path = "/my/custom/path/scheduler/"
```

## 🛠️ Troubleshooting

### Problem: "Failed to load models" Error

**Symptoms**: Error messages about missing model files

**Solutions**:
1. Verify directory structure matches expected format
2. Check file permissions (models should be readable)
3. Ensure all required files are downloaded
4. Verify paths are correct in node configuration

### Problem: "Path not found" Errors

**Symptoms**: Model paths not resolving correctly

**Solutions**:
1. Use absolute paths instead of relative paths
2. Check for typos in path configuration
3. Ensure directories exist and contain model files
4. Try moving models to ComfyUI models directory

### Problem: Download Failures

**Symptoms**: Network errors or incomplete downloads

**Solutions**:
1. Check internet connection
2. Try alternative download methods
3. Use VPN if behind firewall
4. Download manually from browser if needed

## 📊 Model Sizes

**Expected download sizes**:
- **VAE**: ~2-3 GB
- **Transformer**: ~20-25 GB
- **Scheduler**: ~1 KB (config files only)
- **Total**: ~25-30 GB

Ensure you have sufficient disk space before downloading.

## 🔍 Verification

### Check Download Completeness

```python
# Run this script to verify models are correctly downloaded
import os

def check_model_structure():
    base_path = "models"
    required_dirs = ["vae", "transformer", "scheduler"]
    
    for model_dir in required_dirs:
        model_path = os.path.join(base_path, model_dir)
        if not os.path.exists(model_path):
            print(f"❌ Missing: {model_path}")
            return False
        
        # Check for config files
        config_files = ["config.json", "scheduler_config.json"]
        has_config = any(os.path.exists(os.path.join(model_path, cf)) for cf in config_files)
        
        if not has_config:
            print(f"❌ Missing config in: {model_path}")
            return False
            
        print(f"✅ Found: {model_path}")
    
    print("🎉 All models verified successfully!")
    return True

check_model_structure()
```

### Test Model Loading

```python
# Test if models load correctly
from minimax_mask_node_bmo import MinimaxRemoverBMONode

try:
    node = MinimaxRemoverBMONode()
    node.load_models("models/vae/", "models/transformer/", "models/scheduler/")
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
```

## 🔄 Updates

### Checking for Model Updates

```bash
# Check if newer model versions are available
huggingface-cli repo info zibojia/minimax-remover

# Update to latest version
huggingface-cli download zibojia/minimax-remover --local-dir ./models --local-dir-use-symlinks False --force-download
```

## 💡 Best Practices

### Storage Optimization

1. **Use symlinks** if disk space is limited:
   ```bash
   # Download to shared location
   huggingface-cli download zibojia/minimax-remover --local-dir /shared/models
   
   # Create symlinks in project
   ln -s /shared/models/vae models/vae
   ln -s /shared/models/transformer models/transformer
   ln -s /shared/models/scheduler models/scheduler
   ```

2. **Share models** across multiple ComfyUI installations

3. **Use external storage** for large model files

### Performance Tips

1. **SSD storage** recommended for faster model loading
2. **Local storage** preferred over network drives
3. **Sufficient RAM** (16GB+ recommended) for large models

## 🆘 Support

If you encounter issues:

1. **Check the diagnostic output** in the node logs
2. **Verify file integrity** with checksums if available
3. **Try re-downloading** specific model components
4. **Report issues** with detailed error messages

### Useful Commands

```bash
# Check disk space
df -h

# Check model file sizes
du -sh models/*

# List model contents
ls -la models/vae/
ls -la models/transformer/
ls -la models/scheduler/

# Check file permissions
ls -l models/vae/diffusion_pytorch_model.safetensors
```

---

**Note**: Always ensure you have sufficient disk space and a stable internet connection before starting the download process. The models are large and may take significant time to download depending on your connection speed. 