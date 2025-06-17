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
    
    # Check if models are already downloaded
    required_dirs = ["vae", "transformer", "scheduler"]
    if all(os.path.exists(os.path.join(models_dir, d)) for d in required_dirs):
        print("Minimax models already exist.")
        return

    print("Downloading Minimax Remover models...")
    try:
        # Download using huggingface-cli
        subprocess.run([
            "huggingface-cli", "download",
            "zibojia/minimax-remover",
            "--local-dir", models_dir,
            "--local-dir-use-symlinks", "False"
        ], check=True)
        print("Minimax models downloaded successfully!")
    except subprocess.CalledProcessError:
        print("Error downloading Minimax models. Please download manually using:")
        print("huggingface-cli download zibojia/minimax-remover --local-dir ./models --local-dir-use-symlinks False")
        sys.exit(1)

def main():
    print("Starting model downloads...")
    download_sam_model()
    download_minimax_models()
    print("\nAll models downloaded successfully!")
    print("\nProject structure should now look like this:")
    print("""
    your_project/
    ├── sam_vit_h_4b8939.pth
    ├── models/
    │   ├── vae/
    │   ├── transformer/
    │   └── scheduler/
    ├── minimax_sam_node.py
    └── test_run.py
    """)

if __name__ == "__main__":
    main() 