# 🎯 MiniMax-Remover for ComfyUI

**High-quality video object removal with automatic model management and universal resolution support**

<p align="center">
  <a href="https://huggingface.co/zibojia/minimax-remover"><img alt="Huggingface Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Model-brightgreen"></a>
  <a href="https://github.com/zibojia/MiniMax-Remover"><img alt="Github" src="https://img.shields.io/badge/MiniMaxRemover-github-black"></a>
  <a href="https://arxiv.org/abs/2505.24873"><img alt="arXiv" src="https://img.shields.io/badge/MiniMaxRemover-arXiv-b31b1b"></a>
</p>

---

## 🚀 **Major Improvements & Features**

### ✨ **Latest Enhancements**
- ✅ **Auto-Download Models**: Zero-configuration setup - models download automatically on first use
- ✅ **Universal Resolution Support**: Works with ANY resolution input (480p to 4K+)
- ✅ **Smart VAE Compatibility**: Automatic dimension adjustment for perfect compatibility
- ✅ **OpenCV Conflict Resolution**: No more OpenCV reinstallation issues
- ✅ **DW Preprocessor Compatibility**: Works seamlessly with DWPose and other preprocessors
- ✅ **Tensor Dimension Fixes**: Eliminates all dimension mismatch errors
- ✅ **Professional Quality**: Proper VAE normalization for natural, high-quality results

### 🎮 **Core Features**
- **Fast**: Only 6-12 inference steps, highly optimized
- **Robust**: Handles any mask/video resolution combination automatically
- **Plug-and-Play**: Install and use immediately - no manual setup required
- **Production Ready**: Comprehensive error handling and diagnostics

---

## 📥 **Installation**

### **Method 1: ComfyUI Manager (Recommended)**
1. Open ComfyUI Manager
2. Click "Install Custom Nodes"
3. Search for "MiniMax-Remover"
4. Click Install and restart ComfyUI

### **Method 2: Manual Installation**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/CasterPollux/MiniMax-Remover.git
cd MiniMax-Remover
# Install with CUDA support (recommended)
pip install -r requirements.txt
```

### **Method 3: Automatic Setup Script**
```bash
# Run the setup script for automatic ComfyUI integration
python setup_comfyui_integration_bmo.py
```

---

## 🤖 **Model Management - Auto-Download System**

### **Zero-Configuration Experience (Default)**
When you first use the node:

```
🔍 Checking for MiniMax-Remover models...
📥 Models not found locally. Starting automatic download...
🌐 Downloading MiniMax-Remover models to: ./models
📊 Expected download size: ~25-30 GB
⏳ This may take several minutes depending on your connection...
🎉 Models downloaded successfully!
✅ Processing your video...
```

### **Model Download Details**
- **Total Size**: ~25-30 GB (one-time download)
- **Components**: 
  - VAE (~3GB) - Video encoding/decoding
  - Transformer (~25GB) - Main diffusion model
  - Scheduler (~1KB) - Denoising control
- **Source**: [HuggingFace: zibojia/minimax-remover](https://huggingface.co/zibojia/minimax-remover)

### **Storage Locations (Auto-Detected)**
The system automatically tries these locations in priority order:

#### **Option 1: Custom Node Directory (Default)**
```
ComfyUI/custom_nodes/MiniMax-Remover/models/
├── minimax_vae/          (VAE encoder/decoder)
├── minimax_transformer/  (main diffusion model)
└── minimax_scheduler/    (denoising scheduler)
```

#### **Option 2: ComfyUI Models Directory**
```
ComfyUI/models/
├── minimax_vae/          (MiniMax VAE models)
├── minimax_transformer/  (MiniMax Transformer models)
└── minimax_scheduler/    (MiniMax Scheduler configs)
```

#### **Option 3: User Cache Directory (Fallback)**
```
~/.cache/minimax-remover/models/
├── minimax_vae/
├── minimax_transformer/
└── minimax_scheduler/
```
*Windows: `C:\Users\[Username]\.cache\minimax-remover\models\`*

### **Manual Download Options**
If auto-download fails or you prefer manual control:

```bash
# Option 1: Use built-in script
python download_models.py

# Option 2: Direct HuggingFace download
huggingface-cli download zibojia/minimax-remover --local-dir ./models --local-dir-use-symlinks False
# Then rename folders for clarity:
mv ./models/vae ./models/minimax_vae
mv ./models/transformer ./models/minimax_transformer  
mv ./models/scheduler ./models/minimax_scheduler

# Option 3: Download to ComfyUI models directory  
huggingface-cli download zibojia/minimax-remover --local-dir ComfyUI/models --local-dir-use-symlinks False
# Then rename folders for clarity:
mv ComfyUI/models/vae ComfyUI/models/minimax_vae
mv ComfyUI/models/transformer ComfyUI/models/minimax_transformer
mv ComfyUI/models/scheduler ComfyUI/models/minimax_scheduler
```

**📋 For detailed setup instructions, see [`AUTO_DOWNLOAD_GUIDE.md`](./AUTO_DOWNLOAD_GUIDE.md)**

---

## 🎯 **Universal Resolution Support**

### **ANY Resolution Now Supported!**
The node now handles **any resolution input** with **automatic compatibility**:

- ✅ **Any input resolution** - 480p, 720p, 1080p, 4K, custom resolutions
- ✅ **Mixed mask/image sizes** - automatically handles different resolution inputs  
- ✅ **VAE compatibility** - dimensions automatically adjusted for perfect compatibility
- ✅ **No configuration needed** - works automatically

### **Example Resolution Fixes**
| Input Resolution | Problem | Auto-Fixed To | Result |
|------------------|---------|---------------|--------|
| `1000x1778` | 125x222.25 latent | `1008x1784` | 126x223 ✅ |
| `720x480` | No issue | `720x480` | 90x60 ✅ |
| `1920x1080` | 240x135 | `1920x1088` | 240x136 ✅ |

**📋 For technical details, see [`TENSOR_DIMENSION_FIX_GUIDE.md`](./TENSOR_DIMENSION_FIX_GUIDE.md)**

---

## 🛠️ **Compatibility Fixes**

### **OpenCV Conflict Resolution ✅**

**Problem Solved**: Users were experiencing OpenCV reinstallation prompts and conflicts.

**Root Cause**: Multiple conflicting OpenCV version requirements across different files:
- `requirements.txt`: `opencv-python==4.5.5.64` (exact old version)
- `requirements_bmo.txt`: `opencv-python>=4.8.0` (newer minimum)
- `pyproject.toml`: `opencv-python>=4.7.0` (different minimum)

**Solution Applied**:
- ✅ **Standardized OpenCV requirements** to `opencv-python>=4.5.0,<5.0.0` across all files
- ✅ **Safe installation procedures** that don't break existing OpenCV installations
- ✅ **Recovery scripts** for users with broken installations

**Result**: No more OpenCV reinstallation prompts or conflicts!

### **DW Preprocessor Compatibility ✅**

**Problem Solved**: Conflicts between MiniMax-Remover and DWPose/DW preprocessors.

**Root Cause**: Multiple compatibility issues:
- PyTorch version conflicts (`torch==2.6` vs DWPose requirement `<2.4`)
- Segment Anything model loading conflicts
- ONNX runtime conflicts
- CUDA memory issues from simultaneous GPU usage

**Solution Applied**:
- ✅ **Compatible PyTorch range**: `torch>=2.0.0,<2.5.0` for DWPose compatibility
- ✅ **TorchVision compatibility**: Proper version alignment
- ✅ **Memory management**: Smart GPU memory handling
- ✅ **Model isolation**: Prevents loading conflicts

**Result**: MiniMax-Remover now works seamlessly with DWPose and other preprocessors!

**📋 For detailed compatibility info, see [`DW_PREPROCESSOR_COMPATIBILITY.md`](./DW_PREPROCESSOR_COMPATIBILITY.md)**

---

## 🚀 **Usage**

### **In ComfyUI**

1. **Add the Node**: Look for "MiniMax-Remover (BMO)" in the MiniMax-Remover category

2. **Connect Inputs**:
   - `images`: Your video frames as IMAGE type
   - `masks`: Your binary masks as MASK type

3. **Configure (Optional)**:
   - `num_inference_steps`: 12 (default, optimal quality/speed balance)
   - `iterations`: 6 (mask expansion, default works for most cases)
   - `seed`: Any number for reproducible results
   - `auto_download`: True (default, enables automatic model management)

4. **Run**: Perfect results with any resolution combination!

### **Node Parameters**

#### **Core Settings**
- **`auto_download`**: `True` (default) - Automatic model management
- **`num_inference_steps`**: `12` (default) - Quality/speed balance
- **`iterations`**: `6` (default) - Mask expansion iterations
- **`seed`**: `42` (default) - Random seed for reproducible results

#### **Advanced Settings (Auto-Detected)**
- **`vae_path`**: `"auto"` (default) - Auto-detected VAE path
- **`transformer_path`**: `"auto"` (default) - Auto-detected Transformer path  
- **`scheduler_path`**: `"auto"` (default) - Auto-detected Scheduler path

### **What You'll See**

#### **First Use (Auto-Download)**
```
🔧 Using BMO MiniMax-Remover Pipeline (Flexible Resolution)
📐 Input Analysis:
   Images: torch.Size([165, 1920, 1080, 3])
   Masks: torch.Size([165, 1024, 576, 1])
   Using input dimensions as target: 1920x1080

🔧 Auto-resizing for compatibility to 1920x1080
   VAE-compatible target: 1920x1088
   Resizing masks: 1024x576 -> 1920x1088

🎉 Models downloaded successfully with descriptive names!
📁 MiniMax VAE: D:\ComfyUI\custom_nodes\MiniMax-Remover\models\minimax_vae
📁 MiniMax Transformer: D:\ComfyUI\custom_nodes\MiniMax-Remover\models\minimax_transformer
📁 MiniMax Scheduler: D:\ComfyUI\custom_nodes\MiniMax-Remover\models\minimax_scheduler
✅ Processing complete - perfect results!
```

#### **Subsequent Uses (Instant)**
```
🔍 Checking for MiniMax-Remover models...
✅ Found existing models at: D:\ComfyUI\custom_nodes\MiniMax-Remover\models (descriptive names)
   VAE: D:\ComfyUI\custom_nodes\MiniMax-Remover\models\minimax_vae
   Transformer: D:\ComfyUI\custom_nodes\MiniMax-Remover\models\minimax_transformer
   Scheduler: D:\ComfyUI\custom_nodes\MiniMax-Remover\models\minimax_scheduler
✅ BMO MiniMax-Remover models loaded successfully!
🚀 Processing your video...
```


---

## 🔧 **Troubleshooting**

### **Common Issues & Solutions**

#### **Models Not Downloading**
```bash
# Check internet connection and try manual download
python download_models.py

# Or use HuggingFace CLI
huggingface-cli download zibojia/minimax-remover --local-dir ./models
```

#### **PyTorch CUDA Issues**
```bash
# If PyTorch gets downgraded to CPU version during installation:
# 1. Reinstall PyTorch with CUDA support first
pip install torch>=2.0.0,<2.8.0 torchvision>=0.15.0,<0.20.0 --index-url https://download.pytorch.org/whl/cu121

# 2. Then install other dependencies
pip install -r requirements.txt --no-deps --force-reinstall diffusers transformers accelerate
```

#### **OpenCV Conflicts**
```bash
# Install with compatible versions
pip install -r requirements.txt

# If issues persist, see INSTALLATION_GUIDE_OPENCV_FIX.md
```

#### **DW Preprocessor Conflicts**
```bash
# Use compatible PyTorch version
pip install "torch>=2.0.0,<2.5.0" "torchvision>=0.15.0,<0.20.0"

# See DW_PREPROCESSOR_COMPATIBILITY.md for details
```

#### **Memory Issues**
- Use smaller input resolutions
- Enable model offloading in ComfyUI settings
- Close other GPU-intensive applications

#### **Poor Quality Results**
- Ensure you're using the latest BMO version
- Try different seeds (use seed parameter)
- Adjust mask expansion (iterations parameter: 6-10)
- Check that masks are clean and binary

---

## 📁 **Project Structure**

The MiniMax-Remover models can be stored in two different locations. Check both locations to find your models:

```
MiniMax-Remover/
├── 📄 README.md (this file)
├── 🔧 minimax_mask_node_bmo.py (main ComfyUI node)
├── 🔧 pipeline_minimax_remover_bmo.py (processing pipeline)
├── 🔧 transformer_minimax_remover.py (model architecture)
├── 📦 requirements.txt (dependencies)
├── 📁 models/ (auto-downloaded models with descriptive names)
│   ├── minimax_vae/ (VAE encoder/decoder)
│   ├── minimax_transformer/ (main diffusion model)
│   └── minimax_scheduler/ (denoising scheduler)
├── 📋 Documentation/
│   ├── AUTO_DOWNLOAD_GUIDE.md
│   ├── TENSOR_DIMENSION_FIX_GUIDE.md
│   ├── DW_PREPROCESSOR_COMPATIBILITY.md
│   ├── INSTALLATION_GUIDE_OPENCV_FIX.md
│   └── OPENCV_FIX_SUMMARY.md
└── 🛠️ Scripts/
    ├── download_models.py
    ├── setup_comfyui_integration_bmo.py
    └── fix_comfyui_diffusers.py
```

### **Option 2: ComfyUI Models Directory Structure (Alternative)**
```
ComfyUI/
├── models/
│   ├── checkpoints/ (Stable Diffusion models)
│   ├── vae/ (Standard VAE models)  
│   ├── loras/ (LoRA models)
│   ├── minimax_vae/ (🎯 MiniMax VAE - auto-downloaded here)
│   ├── minimax_transformer/ (🎯 MiniMax Transformer - auto-downloaded here)
│   └── minimax_scheduler/ (🎯 MiniMax Scheduler - auto-downloaded here)
└── custom_nodes/
    └── MiniMax-Remover/
        ├── 📄 README.md
        ├── 🔧 minimax_mask_node_bmo.py
        ├── 🔧 pipeline_minimax_remover_bmo.py
        └── 📦 requirements.txt
```

**💡 Tip**: When the node runs, it will display the **actual local paths** where your models are found, so you'll know exactly which location is being used.

---

## 📚 **Documentation**

### **Complete Guides Available**
- **[`AUTO_DOWNLOAD_GUIDE.md`](./AUTO_DOWNLOAD_GUIDE.md)** - Model download and management
- **[`TENSOR_DIMENSION_FIX_GUIDE.md`](./TENSOR_DIMENSION_FIX_GUIDE.md)** - Resolution compatibility details
- **[`DW_PREPROCESSOR_COMPATIBILITY.md`](./DW_PREPROCESSOR_COMPATIBILITY.md)** - DWPose integration guide
- **[`INSTALLATION_GUIDE_OPENCV_FIX.md`](./INSTALLATION_GUIDE_OPENCV_FIX.md)** - OpenCV issue resolution
- **[`OPENCV_FIX_SUMMARY.md`](./OPENCV_FIX_SUMMARY.md)** - Complete fix summary

### **Support Resources**
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive guides for all features
- **Community**: Share workflows and tips



### **Before vs After**

#### **❌ Old Experience**
```
- Manual model download required (25GB)
- Resolution errors with non-standard sizes
- OpenCV conflicts breaking installations  
- DW preprocessor incompatibility
- Complex setup and configuration
```

#### **✅ New Experience**  
```
- Plug-and-play: install and use immediately
- Works with ANY resolution automatically
- No OpenCV conflicts or reinstallations
- Perfect DW preprocessor compatibility
- Professional-quality results every time
```



---

## 🔗 **Links & Resources**

### **Official Project**
- **[Original Paper](https://arxiv.org/abs/2505.24873)** - Technical details and methodology
- **[Official Repository](https://github.com/zibojia/MiniMax-Remover)** - Original implementation
- **[HuggingFace Models](https://huggingface.co/zibojia/minimax-remover)** - Pre-trained models
- **[Demo Page](https://minimax-remover.github.io)** - Live demonstrations

### **ComfyUI Integration**
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** - Main ComfyUI project
- **[ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)** - Node management tool

---

## 📧 **Contact & Support**

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/MiniMax-Remover/issues)
- **Email**: [19210240030@fudan.edu.cn](mailto:19210240030@fudan.edu.cn)
- **Documentation**: All guides included in repository

---

## 📜 **License**

This project is licensed under the terms specified in the [LICENSE](./LICENSE) file.

---

## 🙏 **Credits**

### **Original Authors**
Bojia Zi*, Weixuan Peng*, Xianbiao Qi†, Jianan Wang, Shihao Zhao, Rong Xiao, Kam-Fai Wong  
*Equal contribution. †Corresponding author.

### **ComfyUI Integration & Enhancements**
- Auto-download system implementation
- Universal resolution support
- OpenCV & DW preprocessor compatibility fixes
- Comprehensive documentation and guides

---

**🎨 Happy Video Inpainting! ✨**

*Experience professional-quality video object removal with zero configuration required.*