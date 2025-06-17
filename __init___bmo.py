# BMO MiniMax-Remover Custom Node for ComfyUI
# High-quality video object removal

try:
    from .minimax_mask_node_bmo import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    
    # Export the mappings for ComfyUI
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
    
    print("✅ BMO MiniMax-Remover nodes loaded successfully!")
    
except ImportError as e:
    print(f"❌ Warning: Failed to import BMO MiniMax-Remover nodes: {e}")
    print("   Make sure all dependencies are installed:")
    print("   pip install diffusers transformers torch scipy einops")
    
    # Provide empty mappings as fallback
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 