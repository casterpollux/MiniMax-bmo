#!/usr/bin/env python3
"""
BMO MiniMax-Remover ComfyUI Node
High-quality video object removal - based on official implementation
"""

import torch
import numpy as np
import cv2
from typing import Optional, Union, List
import folder_paths
import comfy.model_management as model_management

# Import the BMO pipeline
import os
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Handle import path for ComfyUI
def get_comfyui_base_path():
    """Get the ComfyUI base path for proper imports"""
    current_dir = Path(__file__).parent.absolute()
    
    # Look for ComfyUI base directory
    comfyui_paths = [
        current_dir.parent.parent,  # custom_nodes/minimax-remover-bmo/
        Path("D:/COMFY_UI/ComfyUI"),  # Direct path
        Path(os.environ.get("COMFYUI_BASE", ""))  # Environment variable
    ]
    
    for path in comfyui_paths:
        if path.exists() and (path / "nodes.py").exists():
            return str(path)
    
    return str(current_dir.parent.parent)

# Add ComfyUI to path
comfyui_base = get_comfyui_base_path()
if comfyui_base not in sys.path:
    sys.path.insert(0, comfyui_base)

try:
    # ComfyUI imports
    import comfy.model_management as model_management
    print("✅ ComfyUI model_management imported successfully")
except ImportError as e:
    print(f"⚠️ Could not import ComfyUI model_management: {e}")
    # Fallback device management
    class MockModelManagement:
        @staticmethod
        def get_torch_device():
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_management = MockModelManagement()

# Import required modules
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler

# Enhanced imports with path handling
current_dir = Path(__file__).parent
try:
    # Try local import first
    sys.path.insert(0, str(current_dir))
    from pipeline_minimax_remover_bmo import Minimax_Remover_Pipeline_BMO
    from transformer_minimax_remover import Transformer3DModel
    print("✅ Local BMO modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import BMO modules: {e}")
    # Attempt to find and import from different locations
    possible_paths = [
        current_dir,
        current_dir.parent,
        Path(comfyui_base) / "custom_nodes" / "minimax-remover-bmo"
    ]
    
    imported = False
    for path in possible_paths:
        try:
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            from pipeline_minimax_remover_bmo import Minimax_Remover_Pipeline_BMO
            from transformer_minimax_remover import Transformer3DModel
            print(f"✅ BMO modules imported from: {path}")
            imported = True
            break
        except ImportError:
            continue
    
    if not imported:
        raise ImportError("Could not import BMO pipeline modules from any location")


class MinimaxRemoverBMONode:
    """
    BMO MiniMax-Remover Node for ComfyUI
    High-quality video object removal with separate model path inputs
    """
    
    def __init__(self):
        self.pipe = None
        self.device = model_management.get_torch_device()
        self.comfyui_models_path = Path(comfyui_base) / "models"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "num_inference_steps": ("INT", {
                    "default": 12, 
                    "min": 6, 
                    "max": 50, 
                    "step": 1,
                    "display": "number"
                }),
                "iterations": ("INT", {
                    "default": 6, 
                    "min": 1, 
                    "max": 20, 
                    "step": 1,
                    "display": "number"
                }),
                "seed": ("INT", {
                    "default": 42, 
                    "min": 0, 
                    "max": 0xffffffffffffffff,
                    "display": "number"
                }),
                "vae_path": ("STRING", {
                    "default": "models/vae/",
                    "multiline": False,
                    "tooltip": "Path to VAE model directory"
                }),
                "transformer_path": ("STRING", {
                    "default": "models/transformer/",
                    "multiline": False,
                    "tooltip": "Path to Transformer model directory"
                }),
                "scheduler_path": ("STRING", {
                    "default": "models/scheduler/",
                    "multiline": False,
                    "tooltip": "Path to Scheduler config directory"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process_video"
    CATEGORY = "MiniMax-Remover"
    DESCRIPTION = "BMO MiniMax video object removal with separate model paths"

    def resolve_model_path(self, model_path: str, model_type: str) -> str:
        """
        Resolve model path with intelligent fallbacks
        
        Args:
            model_path: User-provided path (can be relative or absolute)
            model_type: Type of model (vae, transformer, scheduler)
            
        Returns:
            Absolute path to model directory
        """
        # Convert to Path object
        path = Path(model_path)
        
        # If absolute path and exists, use it
        if path.is_absolute() and path.exists():
            print(f"✅ Using absolute path for {model_type}: {path}")
            return str(path)
        
        # Try relative to ComfyUI base
        comfyui_relative = Path(comfyui_base) / model_path
        if comfyui_relative.exists():
            print(f"✅ Using ComfyUI relative path for {model_type}: {comfyui_relative}")
            return str(comfyui_relative)
        
        # Try in ComfyUI models directory
        models_path = self.comfyui_models_path / model_type
        if models_path.exists():
            print(f"✅ Using ComfyUI models path for {model_type}: {models_path}")
            return str(models_path)
        
        # Try original path as-is (fallback)
        print(f"⚠️ Using fallback path for {model_type}: {path}")
        return str(path)

    def load_models(self, vae_path: str, transformer_path: str, scheduler_path: str):
        """Load MiniMax-Remover models from paths"""
        if self.pipe is not None:
            print("ℹ️ Models already loaded, skipping...")
            return  # Already loaded
            
        print("🔄 Loading BMO MiniMax-Remover models...")
        print(f"🗂️ ComfyUI base path: {comfyui_base}")
        print(f"🗂️ ComfyUI models path: {self.comfyui_models_path}")
        
        try:
            # Resolve model paths
            resolved_vae_path = self.resolve_model_path(vae_path, "vae")
            resolved_transformer_path = self.resolve_model_path(transformer_path, "transformer")
            resolved_scheduler_path = self.resolve_model_path(scheduler_path, "scheduler")
            
            print(f"📁 Loading VAE from: {resolved_vae_path}")
            print(f"📁 Loading Transformer from: {resolved_transformer_path}")
            print(f"📁 Loading Scheduler from: {resolved_scheduler_path}")
            
            # Load models from resolved paths
            vae = AutoencoderKLWan.from_pretrained(
                resolved_vae_path, 
                torch_dtype=torch.float16
            )
            transformer = Transformer3DModel.from_pretrained(
                resolved_transformer_path, 
                torch_dtype=torch.float16
            )
            scheduler = UniPCMultistepScheduler.from_pretrained(
                resolved_scheduler_path
            )
            
            # Create the BMO pipeline
            self.pipe = Minimax_Remover_Pipeline_BMO(
                vae=vae,
                transformer=transformer, 
                scheduler=scheduler
            ).to(self.device)
            
            print("✅ BMO MiniMax-Remover models loaded successfully!")
            print(f"   Using device: {self.device}")
            print(f"   VAE: {type(vae).__name__}")
            print(f"   Transformer: {type(transformer).__name__}")
            print(f"   Scheduler: {type(scheduler).__name__}")
            
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            print("🔍 Debug info:")
            print(f"   VAE path exists: {Path(resolved_vae_path).exists()}")
            print(f"   Transformer path exists: {Path(resolved_transformer_path).exists()}")
            print(f"   Scheduler path exists: {Path(resolved_scheduler_path).exists()}")
            
            # List available files for debugging
            for name, path in [("VAE", resolved_vae_path), ("Transformer", resolved_transformer_path), ("Scheduler", resolved_scheduler_path)]:
                if Path(path).exists():
                    files = list(Path(path).glob("*"))
                    print(f"   {name} directory contents: {[f.name for f in files]}")
            
            raise e

    def process_video(
        self, 
        images, 
        masks, 
        num_inference_steps=12,
        iterations=6,
        seed=42,
        vae_path="models/vae/",
        transformer_path="models/transformer/",
        scheduler_path="models/scheduler/"
    ):
        """
        Process video with BMO MiniMax-Remover
        
        Args:
            images: Input video frames [B, H, W, C] in [0, 1]
            masks: Binary masks [B, H, W] in [0, 1] 
            num_inference_steps: Number of denoising steps (official default: 12)
            iterations: Mask expansion iterations (official default: 6)
            seed: Random seed for reproducible results
            vae_path: Path to VAE model directory
            transformer_path: Path to Transformer model directory
            scheduler_path: Path to Scheduler config directory
            
        Returns:
            Processed video frames [B, H, W, C] in [0, 1]
        """
        
        # Load models
        self.load_models(vae_path, transformer_path, scheduler_path)
        
        print("🚀 Running BMO MiniMax-Remover")
        print("=" * 50)
        
        # Convert ComfyUI tensors to the format expected by MiniMax
        batch_size, height, width, channels = images.shape
        num_frames = batch_size
        
        print(f"📐 Input: {images.shape} frames, {masks.shape} masks")
        print(f"🎯 Parameters: steps={num_inference_steps}, iterations={iterations}, seed={seed}")
        print(f"🗂️ Model paths: VAE={vae_path}, Transformer={transformer_path}, Scheduler={scheduler_path}")
        
        # Prepare images: ComfyUI [B, H, W, C] -> MiniMax [F, H, W, C] -> [-1, 1]
        images_np = images.detach().cpu().numpy()  # [B, H, W, C] in [0, 1]
        images_minimax = images_np * 2.0 - 1.0     # Convert to [-1, 1] for MiniMax
        
        # Prepare masks: ComfyUI [B, H, W] -> MiniMax [F, H, W, C]
        if len(masks.shape) == 3:  # [B, H, W]
            masks_np = masks.detach().cpu().numpy()
            masks_minimax = np.expand_dims(masks_np, axis=-1)  # [F, H, W, 1]
        else:  # [B, H, W, C]
            masks_minimax = masks.detach().cpu().numpy()
        
        print(f"🔄 Converted to MiniMax format:")
        print(f"   Images: {images_minimax.shape} [{images_minimax.min():.3f}, {images_minimax.max():.3f}]")
        print(f"   Masks: {masks_minimax.shape} [{masks_minimax.min():.3f}, {masks_minimax.max():.3f}]")
        
        # Convert to tensors
        images_tensor = torch.from_numpy(images_minimax).float()
        masks_tensor = torch.from_numpy(masks_minimax).float()
        
        # Set up generator for reproducible results
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Run the BMO pipeline
        print(f"🔥 Processing with BMO pipeline...")
        
        try:
            with torch.no_grad():
                result = self.pipe(
                    images=images_tensor,
                    masks=masks_tensor,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    iterations=iterations,
                    generator=generator,
                    output_type="np"  # Get numpy output
                )
            
            # Extract frames
            output_frames = result.frames
            print(f"✅ Pipeline completed!")
            print(f"📊 Output: {output_frames.shape} [{output_frames.min():.3f}, {output_frames.max():.3f}]")
            
            # Convert back to ComfyUI format [B, H, W, C]
            if len(output_frames.shape) == 5:  # [1, F, H, W, C]
                output_frames = output_frames[0]  # Remove batch dimension
            
            # Ensure we have the right number of frames
            if output_frames.shape[0] != num_frames:
                print(f"🔧 Adjusting frame count: {output_frames.shape[0]} -> {num_frames}")
                if output_frames.shape[0] < num_frames:
                    # Pad by repeating last frame
                    last_frame = output_frames[-1:].repeat(num_frames - output_frames.shape[0], axis=0)
                    output_frames = np.concatenate([output_frames, last_frame], axis=0)
                else:
                    # Truncate to desired number of frames
                    output_frames = output_frames[:num_frames]
            
            # Ensure correct shape and range
            output_frames = np.clip(output_frames, 0.0, 1.0)
            
            print(f"📤 Final output: {output_frames.shape} [{output_frames.min():.3f}, {output_frames.max():.3f}]")
            
            # Convert back to tensor for ComfyUI
            result_tensor = torch.from_numpy(output_frames).float()
            
            return (result_tensor,)
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            # Return original images as fallback
            return (images,)


# ComfyUI Node Mappings
NODE_CLASS_MAPPINGS = {
    "MinimaxRemoverBMO": MinimaxRemoverBMONode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimaxRemoverBMO": "MiniMax-Remover (BMO)",
}

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 