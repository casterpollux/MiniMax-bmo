# MiniMax-Remover BMO - ComfyUI Integration

🎉 **High-Quality Video Object Removal** 

This is the **BMO** implementation of MiniMax-Remover for ComfyUI that delivers natural, high-quality inpainting results. Based on the official MiniMax-Remover implementation with proper VAE normalization and dimension handling.

## 🔥 Key Features

- ✅ **Proper VAE Normalization**: Uses official `latents_mean` and `latents_std` from VAE config
- ✅ **Optimized Scheduler**: Properly configured UniPCMultistepScheduler for flow prediction
- ✅ **Dimension Compatibility**: Perfect VAE output vs transformer input alignment
- ✅ **Official Parameters**: Uses optimal defaults (12 steps, 6 iterations)
- ✅ **Natural Results**: Produces solid, realistic inpainting with clean edges

## 📥 Installation

***CLONE THE REPO***

### Method 1: Automatic Setup (Recommended)

1. Run the setup script: this will move all the files you need directly into you comfy ui custom nodes section for you. 
```bash
python setup_comfyui_integration_bmo.py
```

2. Follow the prompts to specify your ComfyUI path
3. Restart ComfyUI completely

### Method 2: Manual Installation

1. Copy these files to your ComfyUI custom_nodes directory:
```
ComfyUI/custom_nodes/minimax-remover-bmo/
├── __init__.py
├── minimax_mask_node_bmo.py
├── pipeline_minimax_remover_bmo.py
└── transformer_minimax_remover.py
└──Models
   ├── vae/
   │   ├── config.json
   │   └── diffusion_pytorch_model.safetensors
   ├── transformer/
   │   ├── config.json
   │   └── diffusion_pytorch_model.safetensors
   └── scheduler/
       └── scheduler_config.json

```

2. Restart ComfyUI

## 🚀 Usage

### In ComfyUI:

1. **Add the node**: Look for "MiniMax-Remover (BMO)" in the MiniMax-Remover category

2. **Connect inputs**:
   - `images`: Your video frames as IMAGE type
   - `masks`: Your binary masks as MASK type

3. **Set parameters**:
   - `num_inference_steps`: 12 (official default, good quality/speed balance)
   - `iterations`: 6 (mask expansion iterations, official default)
   - `seed`: Any number for reproducible results
   - `model_path`: Path to your MiniMax models (default: "models/")

4. **Run**: The output will be clean, natural-looking inpainting!

### Model Setup (also shown above with main files)

Place your MiniMax-Remover models in the same custom nodes section as your main files from installation above:
```
ComfyUI/custom_nodes/minimax-remover-bmo/models
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── transformer/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└── scheduler/
    └── scheduler_config.json
```

## 📊 Performance

- **Processing Time**: ~1-2 seconds for 5 frames at 288x528
- **Memory Usage**: Efficient with proper tensor management
- **Quality**: Natural, solid inpainting results

## 🐛 Troubleshooting

### Poor quality results:
- Ensure you're using the BMO node (latest version)
- Try different seeds  
- Adjust mask expansion (iterations parameter)

### Memory issues:
- Use smaller input resolutions
- Enable model offloading in ComfyUI

## 📝 Example Workflow

1. **Load Video**: Use VHS nodes to load your input video
2. **Create Masks**: Use masking tools or load pre-made masks
3. **Process**: Connect to "MiniMax-Remover (BMO)" node
4. **Output**: Save or preview the cleaned video

## 🎯 Best Practices

- **Resolution**: Works best with resolutions divisible by 16
- **Mask Quality**: Clean, binary masks work best
- **Iterations**: 6-10 for most cases, higher for larger objects
- **Steps**: 12 is optimal, 8-20 range depending on quality needs

## 🔗 Links

- [Original MiniMax-Remover Paper](https://arxiv.org/abs/2412.09940)
- [Official Implementation](https://github.com/miraikan-research/MiniMax-Remover)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

---

**Happy inpainting!** 🎨✨ 
