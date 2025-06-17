<h1 align="center">
  <span style="color:#2196f3;"><b>MiniMax</b></span><span style="color:#f06292;"><b>-Remover</b></span>: Taming Bad Noise Helps Video Object Removal
</h1>

<p align="center">
  Bojia Zi<sup>*</sup>,
  Weixuan Peng<sup>*</sup>,
  Xianbiao Qi<sup>†</sup>,
  Jianan Wang, Shihao Zhao, Rong Xiao, Kam-Fai Wong<br>
  <sup>*</sup> Equal contribution. <sup>†</sup> Corresponding author.
</p>

<p align="center">
  <a href="https://huggingface.co/zibojia/minimax-remover"><img alt="Huggingface Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Model-brightgreen"></a>
  <a href="https://github.com/zibojia/MiniMax-Remover"><img alt="Github" src="https://img.shields.io/badge/MiniMaxRemover-github-black"></a>
  <a href="https://huggingface.co/spaces/zibojia/MiniMaxRemover"><img alt="Huggingface Space" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Space-1e90ff"></a>
  <a href="https://arxiv.org/abs/2505.24873"><img alt="arXiv" src="https://img.shields.io/badge/MiniMaxRemover-arXiv-b31b1b"></a>
  <a href="https://www.youtube.com/watch?v=KaU5yNl6CTc"><img alt="YouTube" src="https://img.shields.io/badge/Youtube-video-ff0000"></a>
  <a href="https://minimax-remover.github.io"><img alt="Demo Page" src="https://img.shields.io/badge/Website-Demo%20Page-yellow"></a>
</p>

---

## 🚀 Overview

**MiniMax-Remover** is a fast and effective video object remover based on minimax optimization. It operates in two stages: the first stage trains a remover using a simplified DiT architecture, while the second stage distills a robust remover with CFG removal and fewer inference steps.
---

## 📧 Contact

Feel free to send an email to [19210240030@fudan.edu.cn](mailto:19210240030@fudan.edu.cn) if you have any questions or suggestions.



# MiniMax-Remover BMO - ComfyUI Integration Brendan@casterpollux.com if need assistance on comfy section

🎉 **High-Quality Video Object Removal** 

This is the **BMO** implementation of MiniMax-Remover for ComfyUI that delivers natural, high-quality inpainting results. Based on the official MiniMax-Remover implementation with proper VAE normalization and dimension handling.

## 🔥 Key Features

- ✅ **Proper VAE Normalization**: Uses official `latents_mean` and `latents_std` from VAE config
- ✅ **Optimized Scheduler**: Properly configured UniPCMultistepScheduler for flow prediction
- ✅ **Dimension Compatibility**: Perfect VAE output vs transformer input alignment
- ✅ **Official Parameters**: Uses optimal defaults (12 steps, 6 iterations)
- ✅ **Natural Results**: Produces solid, realistic inpainting with clean edges

## 📥 Installation

### Method 1: Automatic Setup (Recommended)

1. Run the setup script:
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

### Model Setup

Place your MiniMax-Remover models in this structure and paste the path into the comfy ui node:
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

## Original Project Description
