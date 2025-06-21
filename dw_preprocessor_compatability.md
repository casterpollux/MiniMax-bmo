# 🤝 DW Preprocessor Compatibility Guide

## 🚨 Issue: Conflicts with DWPose/DW Preprocessors

**Problem**: Users report conflicts when using MiniMax-Remover alongside DWPose or other DW preprocessor nodes in ComfyUI.

**Root Causes**:
1. **PyTorch Version Conflicts**: Exact version pins vs. DW preprocessor requirements
2. **Segment Anything Model Conflicts**: Multiple SAM model loading attempts
3. **ONNX Runtime Conflicts**: Different runtime providers and versions
4. **CUDA Memory Issues**: Both nodes are memory-intensive

## ✅ **Fixes Applied**

### **1. PyTorch Version Compatibility**

**Updated Requirements:**
- **Before**: `torch==2.6` (exact pin causing conflicts)
- **After**: `torch>=2.0.0,<2.5.0` (compatible range)

**Compatible with:**
- DWPose preprocessors requiring `torch>=1.13.0,<2.4`
- Most ComfyUI ControlNet preprocessors
- ONNX runtime requirements

### **2. Memory Management**

**Add to your workflow:**
- Use only one pose estimation node at a time
- Clear CUDA cache between different preprocessors
- Consider using CPU fallback for one of the nodes

## 🛠️ **Troubleshooting Steps**

### **If DW Preprocessor Fails After Installing MiniMax-Remover**

1. **Check PyTorch Compatibility:**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   # Should show 2.0.x - 2.4.x range
   ```

2. **Clear Model Cache:**
   ```bash
   # In ComfyUI directory
   rm -rf models/preprocessors/*dwpose*
   # Let DW preprocessor re-download models
   ```

3. **Restart ComfyUI Completely:**
   - Close ComfyUI
   - Clear CUDA cache
   - Restart ComfyUI

### **If Both Nodes Load But Fail During Processing**

**Error**: `CUDA out of memory` or `cannot unpack non-iterable NoneType object`

**Solutions:**

1. **Use Nodes Sequentially (Not Simultaneously):**
   ```
   Image → DW Preprocessor → [Clear Cache] → MiniMax-Remover
   ```

2. **Enable CPU Fallback for DW Preprocessor:**
   - Set DW preprocessor to use CPU mode if available
   - This reduces CUDA memory pressure

3. **Reduce Batch Sizes:**
   - Process single images instead of batches
   - Use lower resolution settings

### **Memory Management Workflow**

```python
# Example workflow order:
1. Load Image
2. Run DW Preprocessor (pose estimation)
3. Clear CUDA cache: torch.cuda.empty_cache()
4. Run MiniMax-Remover (object removal)
5. Clear CUDA cache again
```

## 🔧 **Advanced Compatibility Settings**

### **1. Environment Variables**

Add to your ComfyUI startup:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1
```

### **2. Model Loading Strategy**

**For DW Preprocessor:**
- Use ONNX models instead of TorchScript when possible
- Enable FP16 precision to reduce memory usage

**For MiniMax-Remover:**
- Use `torch.float16` for VAE operations
- Enable gradient checkpointing if available

### **3. Installation Order**

**Recommended installation sequence:**

1. Install DW Preprocessor first
2. Test DW Preprocessor functionality
3. Install MiniMax-Remover (with updated requirements)
4. Restart ComfyUI
5. Test both nodes separately, then together

## 📊 **Compatibility Matrix**

| DW Preprocessor Version | MiniMax-Remover | PyTorch Range | Status |
|------------------------|-----------------|---------------|---------|
| DWPose (latest) | v1.0+ (fixed) | 2.0.0-2.4.9 | ✅ Compatible |
| OpenPose preprocessor | v1.0+ (fixed) | 2.0.0-2.4.9 | ✅ Compatible |
| DWPose TensorRT | v1.0+ (fixed) | 2.0.0-2.4.9 | ⚠️ Test needed |

## 🆘 **Common Error Solutions**

### **Error**: `TypeError: cannot unpack non-iterable NoneType object`
**Cause**: Segment Anything model loading conflict
**Solution**: 
```bash
# Remove conflicting SAM models
rm ComfyUI/models/sam/*
# Restart ComfyUI and let each node download its own SAM model
```

### **Error**: `CUDA error: out of memory`
**Cause**: Both nodes trying to use GPU simultaneously
**Solutions**:
1. Use CPU mode for DW preprocessor
2. Process sequentially with cache clearing
3. Reduce image resolution

### **Error**: `ModuleNotFoundError: No module named 'onnxruntime'`
**Cause**: ONNX runtime version mismatch
**Solution**:
```bash
pip install onnxruntime-gpu==1.15.1 --force-reinstall
```

## 🎯 **Best Practices**

1. **Test Separately First**: Verify each node works independently
2. **Sequential Processing**: Don't run both nodes simultaneously on the same image
3. **Memory Management**: Clear CUDA cache between intensive operations
4. **Version Pinning**: Use the updated requirements files with compatible ranges
5. **Model Isolation**: Let each node manage its own model downloads

## 📝 **Reporting Issues**

If you still experience conflicts after following this guide:

1. **Include PyTorch version**: `python -c "import torch; print(torch.__version__)"`
2. **Include CUDA version**: `nvidia-smi`
3. **Include exact error message** and traceback
4. **Specify DW preprocessor variant** (DWPose, OpenPose, etc.)
5. **Hardware specs** (GPU model, VRAM amount)

---

**Note**: These fixes ensure MiniMax-Remover works alongside DW preprocessors without breaking existing functionality. 
