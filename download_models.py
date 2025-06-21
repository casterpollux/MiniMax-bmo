import os
import subprocess
import sys
from huggingface_hub import snapshot_download

def download_sam_model():
    """Download SAM model if not already present"""
    sam_path = "sam_vit_h_4b8939.pth"
    if not os.path.exists(sam_path):
        print("Downloading SAM model...")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        try:
            subprocess.run(["wget", url, "-O", sam_path], check=True)
            print("SAM model downloaded successfully!")
        except subprocess.CalledProcessError:
            print("Error downloading SAM model. Please download manually from:")
            print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
            sys.exit(1)
    else:
        print("SAM model already exists.")

def download_minimax_models():
    """Download Minimax Remover models if not already present"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Check if models are already downloaded (try descriptive names first)
    descriptive_dirs = ["minimax_vae", "minimax_transformer", "minimax_scheduler"]
    legacy_dirs = ["vae", "transformer", "scheduler"]
    
    if all(os.path.exists(os.path.join(models_dir, d)) for d in descriptive_dirs):
        print("Minimax models already exist (descriptive names).")
        return
    elif all(os.path.exists(os.path.join(models_dir, d)) for d in legacy_dirs):
        print("Minimax models already exist (legacy names).")
        return

    print("Downloading Minimax Remover models...")
    try:
        # Download to temporary location first
        temp_dir = os.path.join(models_dir, "_temp_download")
        subprocess.run([
            "huggingface-cli", "download",
            "zibojia/minimax-remover",
            "--local-dir", temp_dir,
            "--local-dir-use-symlinks", "False"
        ], check=True)
        
        # Rename to descriptive folder names
        import shutil
        
        rename_map = {
            "vae": "minimax_vae",
            "transformer": "minimax_transformer", 
            "scheduler": "minimax_scheduler"
        }
        
        for old_name, new_name in rename_map.items():
            old_path = os.path.join(temp_dir, old_name)
            new_path = os.path.join(models_dir, new_name)
            
            if os.path.exists(old_path):
                if os.path.exists(new_path):
                    shutil.rmtree(new_path)
                shutil.move(old_path, new_path)
                print(f"📁 Moved {old_name} to {new_name}: {os.path.abspath(new_path)}")
        
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
        print("Minimax models downloaded successfully with descriptive names!")
    except subprocess.CalledProcessError:
        print("Error downloading Minimax models. Please download manually using:")
        print("huggingface-cli download zibojia/minimax-remover --local-dir ./models --local-dir-use-symlinks False")
        print("Then rename folders: vae->minimax_vae, transformer->minimax_transformer, scheduler->minimax_scheduler")
        sys.exit(1)

def main():
    print("Starting model downloads...")
    download_sam_model()
    download_minimax_models()
    print("\nAll models downloaded successfully!")
    print("\nProject structure should now look like this:")
    print("""
    MiniMax-Remover/
    ├── sam_vit_h_4b8939.pth
    ├── models/
    │   ├── minimax_vae/ (VAE encoder/decoder)
    │   ├── minimax_transformer/ (main diffusion model)
    │   └── minimax_scheduler/ (denoising scheduler)
    ├── minimax_mask_node_bmo.py
    └── pipeline_minimax_remover_bmo.py
    """)

if __name__ == "__main__":
    main() 