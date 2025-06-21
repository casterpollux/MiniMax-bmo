#!/usr/bin/env python3
"""
BMO MiniMax-Remover ComfyUI Node
High-quality video object removal - based on official implementation
"""

import os
import sys
import torch
import numpy as np
import cv2
from typing import Optional, Union, List
import folder_paths
import comfy.model_management as model_management

# Import the BMO pipeline
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
    COMFYUI_AVAILABLE = True
except ImportError:
    print(f"⚠️ Could not import ComfyUI model_management")
    COMFYUI_AVAILABLE = False
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

# Lazy imports for diffusers components
def lazy_import_diffusers():
    """Lazy import diffusers components to avoid import issues"""
    global AutoencoderKLWan, UniPCMultistepScheduler, Minimax_Remover_Pipeline_BMO, Transformer3DModel
    
    if 'AutoencoderKLWan' not in globals():
        from diffusers.models import AutoencoderKLWan
        from diffusers.schedulers import UniPCMultistepScheduler
        from pipeline_minimax_remover_bmo import Minimax_Remover_Pipeline_BMO
        from transformer_minimax_remover import Transformer3DModel
    
    return AutoencoderKLWan, UniPCMultistepScheduler, Minimax_Remover_Pipeline_BMO, Transformer3DModel

class MinimaxRemoverBMONode:
    """
    BMO MiniMax-Remover Node for ComfyUI
    High-quality video object removal with separate model path inputs
    """
    
    def __init__(self):
        self.pipe = None
        self.device = model_management.get_torch_device()
        self.comfyui_models_path = Path(comfyui_base) / "models"
        
    def auto_download_models(self, force_download=False):
        """
        Automatically download models if they don't exist
        Returns the paths to the downloaded models
        """
        print("🔍 Checking for MiniMax-Remover models...")
        
        # Standard model locations to try
        possible_locations = [
            # Project directory
            Path("models"),
            # ComfyUI models directory  
            self.comfyui_models_path,
            # User's cache directory
            Path.home() / ".cache" / "minimax-remover"
        ]
        
        # Check if models already exist (try both old and new naming)
        for base_path in possible_locations:
            # Try new descriptive names first
            vae_path = base_path / "minimax_vae"
            transformer_path = base_path / "minimax_transformer" 
            scheduler_path = base_path / "minimax_scheduler"
            
            if (vae_path.exists() and transformer_path.exists() and scheduler_path.exists() and
                (vae_path / "config.json").exists() and 
                (transformer_path / "config.json").exists() and
                (scheduler_path / "scheduler_config.json").exists()):
                
                print(f"✅ Found existing models at: {os.path.abspath(base_path)} (descriptive names)")
                print(f"   VAE: {os.path.abspath(vae_path)}")
                print(f"   Transformer: {os.path.abspath(transformer_path)}")
                print(f"   Scheduler: {os.path.abspath(scheduler_path)}")
                return str(vae_path), str(transformer_path), str(scheduler_path)
            
            # Fallback to old generic names for backward compatibility
            vae_path_old = base_path / "vae"
            transformer_path_old = base_path / "transformer" 
            scheduler_path_old = base_path / "scheduler"
            
            if (vae_path_old.exists() and transformer_path_old.exists() and scheduler_path_old.exists() and
                (vae_path_old / "config.json").exists() and 
                (transformer_path_old / "config.json").exists() and
                (scheduler_path_old / "scheduler_config.json").exists()):
                
                print(f"✅ Found existing models at: {os.path.abspath(base_path)} (legacy names)")
                print(f"   VAE: {os.path.abspath(vae_path_old)}")
                print(f"   Transformer: {os.path.abspath(transformer_path_old)}")
                print(f"   Scheduler: {os.path.abspath(scheduler_path_old)}")
                return str(vae_path_old), str(transformer_path_old), str(scheduler_path_old)
        
        if not force_download:
            print("📥 Models not found locally. Starting automatic download...")
        
        # Choose download location (prefer project directory, fallback to cache)
        download_base = Path("models")
        if not download_base.parent.exists() or not os.access(download_base.parent, os.W_OK):
            download_base = Path.home() / ".cache" / "minimax-remover" / "models"
            print(f"📁 Using cache directory for models: {download_base}")
        
        download_base.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"🌐 Downloading MiniMax-Remover models to: {download_base}")
            print(f"📊 Expected download size: ~25-30 GB")
            print(f"⏳ This may take several minutes depending on your connection...")
            
            # Import huggingface_hub for downloading
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                print("❌ huggingface_hub not found. Installing...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
                from huggingface_hub import snapshot_download
            
            # Download with progress to temporary location first
            temp_download = download_base / "_temp_download"
            snapshot_download(
                repo_id="zibojia/minimax-remover",
                local_dir=str(temp_download),
                local_dir_use_symlinks=False,
                allow_patterns=["vae/*", "transformer/*", "scheduler/*"],
                ignore_patterns=["*.git*", "README.md", "*.txt"]
            )
            
            # Move to descriptive folder names
            import shutil
            
            vae_path = download_base / "minimax_vae"
            transformer_path = download_base / "minimax_transformer"
            scheduler_path = download_base / "minimax_scheduler"
            
            # Move downloaded folders to descriptive names
            if (temp_download / "vae").exists():
                if vae_path.exists():
                    shutil.rmtree(vae_path)
                shutil.move(str(temp_download / "vae"), str(vae_path))
                print(f"📁 Moved VAE to: {vae_path}")
            
            if (temp_download / "transformer").exists():
                if transformer_path.exists():
                    shutil.rmtree(transformer_path)
                shutil.move(str(temp_download / "transformer"), str(transformer_path))
                print(f"📁 Moved Transformer to: {transformer_path}")
            
            if (temp_download / "scheduler").exists():
                if scheduler_path.exists():
                    shutil.rmtree(scheduler_path)
                shutil.move(str(temp_download / "scheduler"), str(scheduler_path))
                print(f"📁 Moved Scheduler to: {scheduler_path}")
            
            # Clean up temp directory
            if temp_download.exists():
                shutil.rmtree(temp_download)
            
            if (vae_path.exists() and transformer_path.exists() and scheduler_path.exists()):
                print("🎉 Models downloaded successfully with descriptive names!")
                print(f"📁 MiniMax VAE: {os.path.abspath(vae_path)}")
                print(f"📁 MiniMax Transformer: {os.path.abspath(transformer_path)}")  
                print(f"📁 MiniMax Scheduler: {os.path.abspath(scheduler_path)}")
                return str(vae_path), str(transformer_path), str(scheduler_path)
            else:
                raise Exception("Download completed but models not found in expected locations")
                
        except Exception as e:
            print(f"❌ Auto-download failed: {e}")
            print("\n🔧 Manual download options:")
            print("1. Run: python download_models.py")
            print("2. Run: huggingface-cli download zibojia/minimax-remover --local-dir ./models")
            print("3. Then rename folders: vae->minimax_vae, transformer->minimax_transformer, scheduler->minimax_scheduler")
            print("4. See MODEL_DOWNLOAD_GUIDE.md for detailed instructions")
            
            # Return default paths so user can manually configure (using descriptive names)
            return "models/minimax_vae/", "models/minimax_transformer/", "models/minimax_scheduler/"

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
            },
            "optional": {
                "auto_download": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically download models if not found"
                }),
                "vae_path": ("STRING", {
                    "default": "auto",
                    "multiline": False,
                    "tooltip": "Path to VAE model directory (auto = automatic detection)"
                }),
                "transformer_path": ("STRING", {
                    "default": "auto",
                    "multiline": False,
                    "tooltip": "Path to Transformer model directory (auto = automatic detection)"
                }),
                "scheduler_path": ("STRING", {
                    "default": "auto",
                    "multiline": False,
                    "tooltip": "Path to Scheduler config directory (auto = automatic detection)"
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
            print(f"✅ Using absolute path for {model_type}: {os.path.abspath(path)}")
            return str(path)
        
        # Try relative to ComfyUI base
        comfyui_relative = Path(comfyui_base) / model_path
        if comfyui_relative.exists():
            print(f"✅ Using ComfyUI relative path for {model_type}: {os.path.abspath(comfyui_relative)}")
            return str(comfyui_relative)
        
        # Try in ComfyUI models directory (descriptive names first)
        models_path_descriptive = self.comfyui_models_path / f"minimax_{model_type}"
        if models_path_descriptive.exists():
            print(f"✅ Using ComfyUI models path for {model_type}: {os.path.abspath(models_path_descriptive)}")
            return str(models_path_descriptive)
        
        # Fallback to generic names for backward compatibility
        models_path = self.comfyui_models_path / model_type
        if models_path.exists():
            print(f"✅ Using ComfyUI models path for {model_type}: {os.path.abspath(models_path)} (legacy)")
            return str(models_path)
        
        # Try original path as-is (fallback)
        print(f"⚠️ Using fallback path for {model_type}: {path}")
        return str(path)

    def load_models(self, vae_path: str, transformer_path: str, scheduler_path: str, auto_download: bool = True):
        """Load MiniMax-Remover models from paths with auto-download support"""
        if self.pipe is not None:
            print("ℹ️ Models already loaded, skipping...")
            return  # Already loaded
            
        print("🔄 Loading BMO MiniMax-Remover models...")
        print(f"🗂️ ComfyUI base path: {comfyui_base}")
        print(f"🗂️ ComfyUI models path: {self.comfyui_models_path}")
        print(f"📍 Current working directory: {os.getcwd()}")
        
        # Handle auto-download and path resolution
        if auto_download and (vae_path == "auto" or transformer_path == "auto" or scheduler_path == "auto"):
            print("🤖 Auto-download mode enabled")
            try:
                auto_vae, auto_transformer, auto_scheduler = self.auto_download_models()
                
                # Use auto-detected paths for "auto" values
                if vae_path == "auto":
                    vae_path = auto_vae
                if transformer_path == "auto":
                    transformer_path = auto_transformer
                if scheduler_path == "auto":
                    scheduler_path = auto_scheduler
                    
                print(f"🎯 Auto-resolved paths:")
                print(f"   VAE: {os.path.abspath(vae_path)}")
                print(f"   Transformer: {os.path.abspath(transformer_path)}")
                print(f"   Scheduler: {os.path.abspath(scheduler_path)}")
                
            except Exception as e:
                print(f"⚠️ Auto-download failed, falling back to manual paths: {e}")
                # Keep original paths if auto-download fails
        
        try:
            # Resolve model paths
            resolved_vae_path = self.resolve_model_path(vae_path, "vae")
            resolved_transformer_path = self.resolve_model_path(transformer_path, "transformer")
            resolved_scheduler_path = self.resolve_model_path(scheduler_path, "scheduler")
            
            print(f"📁 Loading VAE from: {resolved_vae_path}")
            print(f"📁 Loading Transformer from: {resolved_transformer_path}")
            print(f"📁 Loading Scheduler from: {resolved_scheduler_path}")
            
            # Check if paths exist before attempting to load
            missing_paths = []
            for name, path in [("VAE", resolved_vae_path), ("Transformer", resolved_transformer_path), ("Scheduler", resolved_scheduler_path)]:
                if not Path(path).exists():
                    missing_paths.append(f"{name}: {path}")
            
            if missing_paths:
                print(f"❌ Missing model paths:")
                for missing in missing_paths:
                    print(f"   {missing}")
                
                if auto_download:
                    print("🔄 Attempting to re-download missing models...")
                    auto_vae, auto_transformer, auto_scheduler = self.auto_download_models(force_download=True)
                    resolved_vae_path = auto_vae
                    resolved_transformer_path = auto_transformer
                    resolved_scheduler_path = auto_scheduler
                else:
                    raise FileNotFoundError(f"Models not found. Enable auto_download or check paths.")
            
            # Load models from resolved paths
            AutoencoderKLWan, UniPCMultistepScheduler, Minimax_Remover_Pipeline_BMO, Transformer3DModel = lazy_import_diffusers()
            
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
        auto_download=True,
        vae_path="auto",
        transformer_path="auto",
        scheduler_path="auto"
    ):
        """
        Process video with BMO MiniMax-Remover with auto-download support
        
        Args:
            images: Input video frames [B, H, W, C] in [0, 1]
            masks: Binary masks [B, H, W] in [0, 1] 
            num_inference_steps: Number of denoising steps (official default: 12)
            iterations: Mask expansion iterations (official default: 6)
            seed: Random seed for reproducible results
            auto_download: Automatically download models if not found
            vae_path: Path to VAE model directory (auto = automatic detection)
            transformer_path: Path to Transformer model directory (auto = automatic detection)
            scheduler_path: Path to Scheduler config directory (auto = automatic detection)
            
        Returns:
            Processed video frames [B, H, W, C] in [0, 1]
        """
        
        # Load models with auto-download support
        self.load_models(vae_path, transformer_path, scheduler_path, auto_download)
        
        print("🚀 Running BMO MiniMax-Remover")
        print("=" * 50)
        
        # Convert ComfyUI tensors to the format expected by MiniMax
        batch_size, height, width, channels = images.shape
        num_frames = batch_size
        
        print(f"📐 Input: {images.shape} frames, {masks.shape} masks")
        print(f"🎯 Parameters: steps={num_inference_steps}, iterations={iterations}, seed={seed}")
        print(f"🤖 Auto-download: {'enabled' if auto_download else 'disabled'}")
        
        # Show resolved paths
        if vae_path == "auto" or transformer_path == "auto" or scheduler_path == "auto":
            print(f"🗂️ Using auto-detected model paths")
        else:
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

    def diagnose_inputs(self, images, masks):
        """
        Diagnostic function to check input compatibility
        Helps users identify potential dimension issues before processing
        """
        print("🔍 DIAGNOSTIC MODE: Analyzing inputs for compatibility")
        print("=" * 60)
        
        # Analyze images
        if images is None:
            print("❌ ERROR: Images tensor is None")
            return False
        
        print(f"📊 Images analysis:")
        print(f"   Shape: {images.shape}")
        print(f"   Type: {type(images)}")
        print(f"   Data type: {images.dtype if hasattr(images, 'dtype') else 'N/A'}")
        print(f"   Range: [{images.min():.3f}, {images.max():.3f}]" if hasattr(images, 'min') else "")
        
        if len(images.shape) != 4:
            print(f"❌ ERROR: Expected 4D tensor [B, H, W, C], got {len(images.shape)}D")
            return False
        
        batch_size, height, width, channels = images.shape
        print(f"   Frames: {batch_size}")
        print(f"   Resolution: {height}x{width}")
        print(f"   Channels: {channels}")
        
        # Analyze masks
        if masks is None:
            print("❌ ERROR: Masks tensor is None")
            return False
        
        print(f"\n Masks analysis:")
        print(f"   Shape: {masks.shape}")
        print(f"   Type: {type(masks)}")
        print(f"   Data type: {masks.dtype if hasattr(masks, 'dtype') else 'N/A'}")
        print(f"   Range: [{masks.min():.3f}, {masks.max():.3f}]" if hasattr(masks, 'min') else "")
        
        # Check dimension compatibility
        print(f"\n🔧 Compatibility analysis:")
        
        # VAE spatial compatibility
        vae_h = ((height + 7) // 8) * 8
        vae_w = ((width + 7) // 8) * 8
        if height == vae_h and width == vae_w:
            print(f"   ✅ Resolution {height}x{width} is VAE-compatible")
        else:
            print(f"   ⚠️ Resolution {height}x{width} will be adjusted to {vae_h}x{vae_w} for VAE compatibility")
        
        # Temporal compatibility
        vae_scale_factor_temporal = 4  # Default value
        temporal_latent_frames = (batch_size - 1) // vae_scale_factor_temporal + 1
        print(f"   📊 Temporal: {batch_size} frames → {temporal_latent_frames} latent frames")
        
        # Check common issues
        issues = []
        
        if channels != 3:
            issues.append(f"Expected 3 channels (RGB), got {channels}")
        
        if batch_size < 1:
            issues.append(f"Invalid frame count: {batch_size}")
        
        if len(masks.shape) not in [3, 4]:
            issues.append(f"Masks should be 3D [F,H,W] or 4D [F,H,W,1], got {len(masks.shape)}D")
        
        if len(masks.shape) == 3 and masks.shape != (batch_size, height, width):
            issues.append(f"Mask shape {masks.shape} doesn't match image frames")
        
        if len(masks.shape) == 4 and masks.shape != (batch_size, height, width, 1):
            issues.append(f"Mask shape {masks.shape} doesn't match expected [F,H,W,1]")
        
        # Report results
        if issues:
            print(f"\n❌ ISSUES DETECTED:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
            return False
        else:
            print(f"\n✅ ALL CHECKS PASSED - Inputs are compatible!")
            return True


# ComfyUI Node Mappings
NODE_CLASS_MAPPINGS = {
    "MinimaxRemoverBMO": MinimaxRemoverBMONode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimaxRemoverBMO": "MiniMax-Remover (BMO)",
}

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 