# BMO MiniMax-Remover Custom Node for ComfyUI
# High-quality video object removal with separate model path inputs

try:
    from .minimax_mask_node_bmo import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    
    print("✅ BMO MiniMax-Remover node loaded successfully!")
    print("🎯 Available node: MiniMax-Remover (BMO)")
    print("📁 Features: Separate VAE, Transformer, and Scheduler path inputs")
    
    # Export the mappings for ComfyUI
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
    
except ImportError as e:
    print(f"❌ Warning: Failed to import BMO MiniMax-Remover node: {e}")
    print("   Make sure all dependencies are installed:")
    print("   pip install diffusers transformers torch scipy einops")
    
    # Provide empty mappings as fallback
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 