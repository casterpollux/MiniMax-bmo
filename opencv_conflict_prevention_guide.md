# 🔧 MiniMax-Remover Installation Guide - OpenCV Safe Version

## 🚨 Important: OpenCV Conflict Prevention

**This project has been updated to prevent OpenCV/CV2 conflicts that were causing users to reinstall OpenCV.**

### What Was The Problem?

The original project had conflicting OpenCV version requirements:
- `requirements.txt`: `opencv-python==4.5.5.64` (exact old version)
- `requirements_bmo.txt`: `opencv-python>=4.8.0` (newer version)
- `pyproject.toml`: `opencv-python>=4.7.0` (different version)

This caused pip to uninstall and reinstall OpenCV during installation, often breaking existing CV2 installations.

### ✅ What We Fixed

1. **Standardized OpenCV Version**: All requirements now use `opencv-python>=4.5.0,<5.0.0`
2. **Safe Fix Script**: Updated `fix_comfyui_diffusers.py` to check and preserve OpenCV
3. **Compatible Range**: Uses a version range that works with existing installations

## 🚀 Safe Installation Methods

### Method 1: ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Click "Install Custom Nodes"
3. Search for "MiniMax-Remover"
4. Click Install and restart ComfyUI

*Note: Your existing OpenCV installation should remain intact.*

### Method 2: Manual Installation (Safe)
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/MiniMax-Remover.git
cd MiniMax-Remover

# Check if OpenCV is working first
python -c "import cv2; print(f'OpenCV {cv2.__version__} is working')"

# Install with the safe requirements
pip install -r requirements_bmo.txt

# Verify OpenCV still works
python -c "import cv2; print(f'OpenCV {cv2.__version__} still working')"
```

### Method 3: Using the Safe Fix Script

If you encounter issues, use our updated safe fix script:

```bash
python fix_comfyui_diffusers.py
```

**New Features of Safe Fix Script:**
- ✅ Checks OpenCV status before making changes
- ✅ Uses `--no-deps` and `--force-reinstall` instead of uninstalling
- ✅ Automatically restores OpenCV if it gets broken
- ✅ Provides clear status of both diffusers and OpenCV

## 🔍 Verification Steps

After installation, verify everything works:

```bash
# Test OpenCV
python -c "import cv2; print(f'✅ OpenCV {cv2.__version__} working')"

# Test diffusers
python -c "from diffusers.models import AutoencoderKLWan; print('✅ AutoencoderKLWan available')"

# Test the node (in ComfyUI)
# Look for "MiniMax-Remover (BMO)" in the node menu
```

## 🆘 If You Still Have Issues

### Quick OpenCV Restore
If OpenCV gets broken despite our fixes:

```bash
# For ComfyUI embedded Python
path/to/ComfyUI/python_embeded/python.exe -m pip install opencv-python --upgrade

# For system Python
pip install opencv-python --upgrade
```

### Check Installation Status
```bash
# Run our diagnostic
python fix_comfyui_diffusers.py

# This will show:
# ✅ OpenCV status
# ✅ Diffusers status  
# ✅ Automatic fixes if needed
```

## 📋 Technical Details

### OpenCV Version Compatibility
- **Minimum**: `4.5.0` (supports most features used by the project)
- **Maximum**: `<5.0.0` (excludes major version changes)
- **Recommended**: `4.8.0+` (best performance and compatibility)

### Why This Range Works
- Compatible with existing installations (4.5.x - 4.11.x)
- Doesn't force downgrades or upgrades
- Allows pip to choose the best compatible version
- Prevents exact version conflicts

## 🎯 Node Usage

Once installed safely, look for these nodes in ComfyUI:
- **"MiniMax-Remover (BMO)"** - Main processing node
- **"MiniMax-Remover Loader (BMO)"** - Model loader node

## 🔄 Migration from Old Version

If you installed the old version that broke OpenCV:

1. **Restore OpenCV**: `pip install opencv-python --upgrade`
2. **Update the node**: `git pull` in the MiniMax-Remover directory
3. **Run safe fix**: `python fix_comfyui_diffusers.py`
4. **Restart ComfyUI**

---

**Note**: This updated version prioritizes preserving your existing Python environment while ensuring the MiniMax-Remover functionality works correctly. 
