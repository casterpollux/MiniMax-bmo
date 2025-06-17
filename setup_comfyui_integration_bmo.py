#!/usr/bin/env python3
"""
Setup script for integrating BMO MiniMax-Remover into ComfyUI
This script helps you copy the necessary files to your ComfyUI installation
"""

import os
import shutil
import sys
from pathlib import Path

def main():
    print("🚀 MiniMax-Remover BMO ComfyUI Integration Setup")
    print("=" * 60)
    
    # Current directory (where this script is located)
    current_dir = Path(__file__).parent
    
    # Find ComfyUI directory
    comfyui_path = None
    
    # Common ComfyUI locations
    possible_paths = [
        Path.home() / "ComfyUI",
        Path("../ComfyUI"),
        Path("../../ComfyUI"),
        Path("C:/ComfyUI"),
        Path("/opt/ComfyUI"),
    ]
    
    # Ask user for ComfyUI path
    print("Please enter your ComfyUI installation path:")
    print("(or press Enter to auto-detect)")
    user_path = input("ComfyUI path: ").strip()
    
    if user_path:
        comfyui_path = Path(user_path)
    else:
        # Try to auto-detect
        for path in possible_paths:
            if path.exists() and (path / "main.py").exists():
                comfyui_path = path
                print(f"📍 Auto-detected ComfyUI at: {comfyui_path}")
                break
    
    if not comfyui_path or not comfyui_path.exists():
        print("❌ ComfyUI path not found. Please specify the correct path.")
        return False
    
    # Check if it's a valid ComfyUI installation
    if not (comfyui_path / "main.py").exists():
        print("❌ Invalid ComfyUI installation (main.py not found)")
        return False
    
    # Create custom_nodes directory if it doesn't exist
    custom_nodes_dir = comfyui_path / "custom_nodes" / "minimax-remover-bmo"
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Target directory: {custom_nodes_dir}")
    
    # Files to copy
    files_to_copy = [
        "pipeline_minimax_remover_bmo.py",
        "minimax_mask_node_bmo.py", 
        "transformer_minimax_remover.py",
        "__init___bmo.py",
        "requirements_bmo.txt"
    ]
    
    # Copy files
    print("\n📋 Copying files...")
    for file_name in files_to_copy:
        src_file = current_dir / file_name
        dst_file = custom_nodes_dir / file_name
        
        if src_file.exists():
            # Special handling for __init___bmo.py -> __init__.py
            if file_name == "__init___bmo.py":
                dst_file = custom_nodes_dir / "__init__.py"
            # Special handling for requirements_bmo.txt -> requirements.txt
            elif file_name == "requirements_bmo.txt":
                dst_file = custom_nodes_dir / "requirements.txt"
            
            shutil.copy2(src_file, dst_file)
            print(f"✅ Copied: {file_name} -> {dst_file.name}")
        else:
            print(f"⚠️  Warning: {file_name} not found in current directory")
    
    print("✅ All BMO files copied successfully!")
    
    # Create installation instructions
    instructions = f"""
🎉 BMO MiniMax-Remover Integration Complete!

📍 Installation Location: {custom_nodes_dir}

📋 Next Steps:
1. Restart ComfyUI completely
2. Look for the new nodes in the "MiniMax-Remover" category:
   - "MiniMax-Remover (BMO)" - Main processing node
   - "MiniMax-Remover Loader (BMO)" - Model loader node

🔧 Model Setup:
- Place your MiniMax-Remover models in: {comfyui_path}/models/
- Expected structure:
  models/
  ├── vae/
  ├── transformer/
  └── scheduler/

🚀 Usage:
1. Load your video frames as IMAGE
2. Load your masks as MASK  
3. Connect to the "MiniMax-Remover (BMO)" node
4. Set parameters:
   - num_inference_steps: 12 (official default)
   - iterations: 6 (official default)
   - seed: Any number for reproducible results

✨ Key Features:
- Proper VAE normalization for natural results
- Correct dimension handling 
- Official MiniMax parameters
- Clean, efficient processing
- Fast performance (1-2 seconds)

🎯 The BMO pipeline delivers natural, high-quality inpainting results!
"""
    
    print(instructions)
    
    # Save instructions to file
    with open(custom_nodes_dir / "INSTALLATION_COMPLETE.txt", "w") as f:
        f.write(instructions)
    
    print(f"💾 Instructions saved to: {custom_nodes_dir / 'INSTALLATION_COMPLETE.txt'}")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Setup completed successfully!")
        print("Please restart ComfyUI to see the new nodes.")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1) 