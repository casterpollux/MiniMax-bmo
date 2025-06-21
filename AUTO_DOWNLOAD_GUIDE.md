# 🤖 Auto-Download Feature Guide

## Overview
The MiniMax-Remover node now includes **automatic model downloading** and **path detection**, making it incredibly easy to get started without manual setup!

## ✨ Key Features

### 1. **Zero-Configuration Setup**
- Models download automatically when needed
- Paths are detected and populated automatically
- No manual configuration required for basic usage

### 2. **Intelligent Path Detection**
- Searches multiple standard locations for existing models
- Uses the best available location automatically
- Falls back gracefully if auto-download fails

### 3. **Smart Caching**
- Downloads models once and reuses them
- Shares models across multiple ComfyUI installations
- Efficient storage management

## 🚀 How It Works

### Default Experience (Recommended)
When you first use the node:

1. **Auto-Detection**: Node checks for existing models in standard locations
2. **Auto-Download**: If models aren't found, automatically downloads them (~25GB)
3. **Auto-Configure**: Paths are set automatically
4. **Ready to Use**: Node loads and processes your video

### Node Parameters

The node interface now includes:

#### **Auto-Download (Enabled by default)**
- ✅ **`auto_download: True`** - Automatic model management
- ❌ **`auto_download: False`** - Manual path configuration only

#### **Path Settings (Auto by default)**
- **`vae_path: "auto"`** - Automatically detected VAE path
- **`transformer_path: "auto"`** - Automatically detected Transformer path  
- **`scheduler_path: "auto"`** - Automatically detected Scheduler path

## 📁 Model Storage Locations

The auto-download system tries these locations in order:

### 1. **Project Directory** (Preferred)
```
MiniMax-Remover/
└── models/
    ├── vae/
    ├── transformer/
    └── scheduler/
```

### 2. **ComfyUI Models Directory**
```
ComfyUI/
└── models/
    ├── vae/
    ├── transformer/
    └── scheduler/
```

### 3. **User Cache Directory**
```
~/.cache/minimax-remover/
└── models/
    ├── vae/
    ├── transformer/
    └── scheduler/
```

## 🎯 Usage Examples

### Example 1: Zero-Configuration (Default)
```python
# Everything is automatic!
result = minimax_node.process_video(
    images=your_video_frames,
    masks=your_masks,
    # All other parameters use smart defaults
)
```

**What happens:**
1. Node detects no models exist
2. Automatically downloads models (~25GB, one-time)
3. Processes your video
4. Future runs use cached models instantly

### Example 2: Custom Paths with Auto-Download
```python
# Use custom paths but still auto-download if needed
result = minimax_node.process_video(
    images=your_video_frames,
    masks=your_masks,
    auto_download=True,
    vae_path="/my/custom/vae/path",
    transformer_path="auto",  # Auto-detect transformer
    scheduler_path="auto"     # Auto-detect scheduler
)
```

### Example 3: Manual Configuration (Traditional)
```python
# Disable auto-download for full manual control
result = minimax_node.process_video(
    images=your_video_frames,
    masks=your_masks,
    auto_download=False,
    vae_path="./models/vae/",
    transformer_path="./models/transformer/",
    scheduler_path="./models/scheduler/"
)
```

## 📊 First-Time Setup Experience

### What You'll See

#### **Step 1: Model Detection**
```
🔍 Checking for MiniMax-Remover models...
📥 Models not found locally. Starting automatic download...
```

#### **Step 2: Automatic Download**
```
🌐 Downloading MiniMax-Remover models to: ./models
📊 Expected download size: ~25-30 GB
⏳ This may take several minutes depending on your connection...
```

#### **Step 3: Download Progress**
The download includes progress indicators and status updates.

#### **Step 4: Auto-Configuration**
```
🎉 Models downloaded successfully!
🎯 Auto-resolved paths:
   VAE: ./models/vae
   Transformer: ./models/transformer
   Scheduler: ./models/scheduler
```

#### **Step 5: Ready to Use**
```
✅ BMO MiniMax-Remover models loaded successfully!
🚀 Running BMO MiniMax-Remover
```

## 🛠️ Advanced Configuration

### Environment Variables
You can set preferred download locations:

```bash
# Set custom cache directory
export MINIMAX_CACHE_DIR="/my/preferred/cache"

# Set custom ComfyUI models path
export COMFYUI_MODELS_PATH="/custom/comfyui/models"
```

### Manual Override
Even with auto-download enabled, you can specify exact paths:

```python
# Mix auto and manual paths
result = minimax_node.process_video(
    images=your_video_frames,
    masks=your_masks,
    auto_download=True,
    vae_path="/fast/ssd/vae/",      # Custom VAE location
    transformer_path="auto",         # Auto-detect transformer
    scheduler_path="/shared/scheduler/" # Custom scheduler location
)
```

## 🔧 Troubleshooting

### Download Issues

#### **Network Problems**
```
❌ Auto-download failed: Network connection error
🔧 Manual download options:
1. Run: python download_models.py
2. Run: huggingface-cli download zibojia/minimax-remover --local-dir ./models
3. See MODEL_DOWNLOAD_GUIDE.md for detailed instructions
```

**Solutions:**
- Check internet connection
- Try downloading manually using provided commands
- Use VPN if behind firewall

#### **Disk Space Issues**
```
❌ Auto-download failed: No space left on device
```

**Solutions:**
- Free up ~30GB disk space
- Download to external drive using custom path
- Use manual download to specific location

#### **Permission Issues**
```
❌ Auto-download failed: Permission denied
📁 Using cache directory for models: /home/user/.cache/minimax-remover/models
```

**Solutions:**
- Auto-switches to user cache directory
- Check file permissions in project directory
- Run with appropriate permissions

### Path Resolution Issues

#### **Models Not Found**
```
⚠️ Auto-download failed, falling back to manual paths
❌ Missing model paths:
   VAE: ./models/vae
```

**Solutions:**
- Enable auto-download: `auto_download=True`
- Check manually downloaded model structure
- Verify model files exist and are complete

## 💡 Best Practices

### For New Users
1. **Use defaults**: Let auto-download handle everything
2. **Stable connection**: Ensure reliable internet for first download
3. **Sufficient space**: Have ~30GB free disk space

### For Power Users
1. **Share models**: Download once, symlink to multiple ComfyUI instances
2. **SSD storage**: Place models on fast storage for better performance
3. **Custom locations**: Use environment variables for system-wide configuration

### For Shared Systems
1. **Central storage**: Download to shared location accessible by all users
2. **Symlinks**: Link to shared models instead of duplicating
3. **Permissions**: Ensure read access for all users

## 🔄 Migration from Manual Setup

If you already have models downloaded manually:

### Option 1: Let Auto-Detection Find Them
- Models in standard locations will be found automatically
- No migration needed

### Option 2: Move to Standard Location
```bash
# Move existing models to standard location
mv /old/path/models ./models
```

### Option 3: Use Custom Paths
- Set paths to your existing model locations
- Disable auto-download if you prefer manual control

## 📈 Performance Benefits

### First Run
- **Download**: One-time ~25GB download
- **Time**: 10-30 minutes depending on connection
- **Storage**: ~30GB total

### Subsequent Runs
- **Loading**: Models load from local cache
- **Time**: ~10-30 seconds for model loading
- **Storage**: No additional space needed

## 🆘 Support

If auto-download isn't working:

1. **Check logs**: Enable verbose logging to see detailed error messages
2. **Try manual**: Use `python download_models.py` as fallback
3. **Check space**: Ensure sufficient disk space
4. **Verify connection**: Test internet connectivity
5. **Report issues**: Include auto-download logs in bug reports

## 🎉 Summary

The auto-download feature makes MiniMax-Remover **plug-and-play**:

- ✅ **Zero configuration** for most users
- ✅ **Automatic model management** 
- ✅ **Intelligent path detection**
- ✅ **Graceful fallbacks** when issues occur
- ✅ **Compatible with existing setups**

Just drag the node into ComfyUI, connect your inputs, and it works! 🚀 