import torch

print("✅ CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("🚀 GPU:", torch.cuda.get_device_name(0))
else:
    print("�� Running on CPU") 