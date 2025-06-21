from typing import Optional, Union, List

import torch
import scipy
import numpy as np
import torch.nn.functional as functional
from einops import rearrange

from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

from transformer_minimax_remover import Transformer3DModel

class Minimax_Remover_Pipeline_BMO(DiffusionPipeline):
    """
    BMO MiniMax-Remover Pipeline - Clean implementation based on official code
    """
    model_cpu_offload_seq = "transformer->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        transformer: Transformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: UniPCMultistepScheduler
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

        # Use correct VAE scale factors from official implementation
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if hasattr(self.vae, "temperal_downsample") else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if hasattr(self.vae, "temperal_downsample") else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 720,
        width: int = 1280,
        num_latent_frames: int = 21,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def expand_masks(self, masks, iterations):
        """Official mask expansion method - exact copy from official code"""
        masks = masks.cpu().detach().numpy()
        # numpy array, masks [0,1], f h w c
        masks2 = []
        for i in range(len(masks)):
            mask = masks[i]
            mask = mask > 0
            mask = scipy.ndimage.binary_dilation(mask, iterations=iterations)
            masks2.append(mask)
        masks = np.array(masks2).astype(np.float32)
        # Convert to 3-channel: expand last dimension
        if len(masks.shape) == 3:  # [f, h, w]
            masks = np.expand_dims(masks, axis=-1)  # [f, h, w, 1]
        masks = np.repeat(masks, 3, axis=-1)  # [f, h, w, 3]
        masks = rearrange(masks, "f h w c -> c f h w")
        masks = masks[None, ...]  # Add batch dimension
        # Convert back to torch tensor
        masks = torch.from_numpy(masks)
        return masks

    def resize(self, images, w, h):
        """Official resize method - exact copy from official code with dimension validation"""
        print(f"🔄 Resizing from {images.shape} to target resolution {h}x{w}")
        
        # Validate input dimensions
        if len(images.shape) != 5:
            raise ValueError(f"Expected 5D tensor [B, C, F, H, W], got {images.shape}")
        
        bsz, channels, frames, orig_h, orig_w = images.shape
        print(f"   Input: batch={bsz}, channels={channels}, frames={frames}, size={orig_h}x{orig_w}")
        
        # Ensure target dimensions are valid
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid target dimensions: {h}x{w}")
        
        # Ensure dimensions are compatible with VAE scale factor
        if h % 8 != 0 or w % 8 != 0:
            print(f"⚠️ Warning: Target dimensions {h}x{w} not divisible by 8 (VAE scale factor)")
            # Round to nearest multiple of 8
            h = ((h + 7) // 8) * 8
            w = ((w + 7) // 8) * 8
            print(f"   → Adjusted to {h}x{w} for VAE compatibility")
        
        # Perform resize operation
        images = rearrange(images, "b c f h w -> (b f) c h w")
        print(f"   Rearranged for interpolation: {images.shape}")
        
        images = functional.interpolate(images, (h, w), mode='bilinear', align_corners=False)
        print(f"   After interpolation: {images.shape}")
        
        images = rearrange(images, "(b f) c h w -> b c f h w", b=bsz)
        print(f"   Final resized: {images.shape}")
        
        return images

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def validate_inputs(self, images, masks, height, width, num_frames):
        """Validate input tensors and dimensions for compatibility with auto-resize"""
        print("🔍 Validating inputs for dimension compatibility...")
        
        # Validate images
        if images is None:
            raise ValueError("Images tensor cannot be None")
        
        if len(images.shape) != 4:
            raise ValueError(f"Expected images shape [F, H, W, C], got {images.shape}")
        
        frames, img_h, img_w, channels = images.shape
        if frames != num_frames:
            print(f"⚠️ Frame count mismatch: images have {frames} frames, expected {num_frames}")
            num_frames = frames  # Use actual frame count
        
        if channels != 3:
            raise ValueError(f"Expected 3 channels for images, got {channels}")
        
        # Validate masks
        if masks is None:
            raise ValueError("Masks tensor cannot be None")
        
        # Check mask dimensions and auto-resize if needed
        mask_frames, mask_h, mask_w = None, None, None
        
        if len(masks.shape) == 3:
            mask_frames, mask_h, mask_w = masks.shape
            print(f"📊 Mask format: [F, H, W] = {masks.shape}")
        elif len(masks.shape) == 4:
            mask_frames, mask_h, mask_w, mask_channels = masks.shape
            if mask_channels != 1:
                raise ValueError(f"Expected 1 channel for masks, got {mask_channels}")
            print(f"📊 Mask format: [F, H, W, C] = {masks.shape}")
        else:
            raise ValueError(f"Expected mask shape [F, H, W] or [F, H, W, 1], got {masks.shape}")
        
        # Check if mask dimensions match image dimensions
        if mask_frames != frames or mask_h != img_h or mask_w != img_w:
            print(f"⚠️ Mask-Image dimension mismatch detected:")
            print(f"   Images: {frames} frames, {img_h}x{img_w}")
            print(f"   Masks:  {mask_frames} frames, {mask_h}x{mask_w}")
            
            # Auto-resize is handled later in the pipeline, just warn here
            if mask_frames != frames:
                print(f"   → Frame count will be adjusted: {mask_frames} → {frames}")
            if mask_h != img_h or mask_w != img_w:
                print(f"   → Mask resolution will be resized: {mask_h}x{mask_w} → {img_h}x{img_w}")
        
        # Validate and adjust dimensions for compatibility
        if height != img_h or width != img_w:
            print(f"📏 Input size {img_h}x{img_w} will be resized to {height}x{width}")
        
        # Check VAE compatibility
        vae_compatible_h = ((height + 7) // 8) * 8
        vae_compatible_w = ((width + 7) // 8) * 8
        
        if height != vae_compatible_h or width != vae_compatible_w:
            print(f"⚠️ Adjusting dimensions for VAE compatibility: {height}x{width} → {vae_compatible_h}x{vae_compatible_w}")
            height, width = vae_compatible_h, vae_compatible_w
        
        # Check temporal compatibility
        if num_frames < 1:
            raise ValueError(f"Invalid number of frames: {num_frames}")
        
        temporal_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        print(f"📊 Temporal processing: {num_frames} frames → {temporal_latent_frames} latent frames")
        
        print(f"✅ Input validation passed:")
        print(f"   Frames: {num_frames}")
        print(f"   Resolution: {height}x{width}")
        print(f"   VAE scale factors: spatial={self.vae_scale_factor_spatial}, temporal={self.vae_scale_factor_temporal}")
        
        return height, width, num_frames

    def auto_resize_for_compatibility(self, images, masks, target_height, target_width):
        """
        Automatically resize images and masks to be compatible with each other and the VAE
        This handles any input resolution combination and ensures VAE compatibility
        """
        print(f"🔧 Auto-resizing for compatibility to {target_height}x{target_width}")
        
        # CRITICAL: Ensure target dimensions are VAE compatible AND even numbers
        # VAE requires multiples of 8, but odd multiples can cause issues in diffusion layers
        original_target = f"{target_height}x{target_width}"
        
        # Round to nearest multiple of 8
        target_height = ((target_height + 7) // 8) * 8
        target_width = ((target_width + 7) // 8) * 8
        
        # Ensure even multiples of 8 for maximum compatibility (avoid 8, 24, 40, etc.)
        # If the result is an odd multiple of 8, round to the next even multiple
        height_factor = target_height // 8
        width_factor = target_width // 8
        
        if height_factor % 2 == 1:  # Odd multiple of 8
            target_height = (height_factor + 1) * 8
            print(f"   Adjusted height to even multiple: {height_factor}*8 -> {(height_factor + 1)}*8")
            
        if width_factor % 2 == 1:  # Odd multiple of 8  
            target_width = (width_factor + 1) * 8
            print(f"   Adjusted width to even multiple: {width_factor}*8 -> {(width_factor + 1)}*8")
        
        print(f"   VAE-compatible target: {original_target} -> {target_height}x{target_width}")
        print(f"   Latent dimensions will be: {target_height//8}x{target_width//8}")
        
        # Process images
        print(f"   Original images: {images.shape}")
        if len(images.shape) == 4:  # [F, H, W, C]
            images = rearrange(images, "f h w c -> c f h w")  # -> [C, F, H, W]
            images = images.unsqueeze(0)  # -> [1, C, F, H, W]
        
        # Resize images if needed
        _, _, _, img_h, img_w = images.shape
        if img_h != target_height or img_w != target_width:
            print(f"   Resizing images: {img_h}x{img_w} -> {target_height}x{target_width}")
            images = self.resize(images, target_width, target_height)
        
        # Process masks
        print(f"   Original masks: {masks.shape}")
        
        # Handle expanded mask format from expand_masks: [1, 3, F, H, W]
        if len(masks.shape) == 5:  # Already processed by expand_masks
            _, _, mask_frames, mask_h, mask_w = masks.shape
        elif len(masks.shape) == 4 and masks.shape[-1] == 1:
            masks = masks.squeeze(-1)  # [F, H, W, 1] -> [F, H, W]
            mask_frames, mask_h, mask_w = masks.shape
        elif len(masks.shape) == 3:
            mask_frames, mask_h, mask_w = masks.shape
        else:
            raise ValueError(f"Unexpected mask shape: {masks.shape}")
        
        # Resize masks if needed
        if mask_h != target_height or mask_w != target_width:
            print(f"   Resizing masks: {mask_h}x{mask_w} -> {target_height}x{target_width}")
            
            if len(masks.shape) == 5:  # Expanded masks [1, 3, F, H, W]
                masks = self.resize(masks, target_width, target_height)
            else:  # Regular masks [F, H, W]
                # Convert masks to 5D for resize: [F, H, W] -> [1, 1, F, H, W]
                masks_5d = masks.unsqueeze(0).unsqueeze(0)
                masks_5d = self.resize(masks_5d, target_width, target_height)
                masks = masks_5d.squeeze(0).squeeze(0)  # -> [F, H, W]
        
        print(f"✅ Compatibility check complete:")
        print(f"   Images: {images.shape}")
        print(f"   Masks: {masks.shape}")
        print(f"   Final resolution: {target_height}x{target_width} (latent: {target_height//8}x{target_width//8})")
        
        return images, masks, target_height, target_width

    @torch.no_grad()
    def __call__(
        self,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        images: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        iterations: int = 16
    ):
        """
        BMO inference method - supports any resolution input with automatic compatibility
        """
        print("🔧 Using BMO MiniMax-Remover Pipeline (Flexible Resolution)")
        
        self._current_timestep = None
        self._interrupt = False
        device = self._execution_device
        batch_size = 1
        transformer_dtype = torch.float16

        # Get actual input dimensions
        if len(images.shape) == 4:  # [F, H, W, C]
            input_frames, input_height, input_width, input_channels = images.shape
        else:
            raise ValueError(f"Expected images shape [F, H, W, C], got {images.shape}")
        
        print(f"📐 Input Analysis:")
        print(f"   Images: {images.shape}")
        print(f"   Masks: {masks.shape}")
        print(f"   Requested: {height}x{width}")
        print(f"   Actual input: {input_height}x{input_width}")
        
        # Use input dimensions if no specific target is requested
        if height == 720 and width == 1280:  # Default values
            target_height, target_width = input_height, input_width
            print(f"   Using input dimensions as target: {target_height}x{target_width}")
        else:
            target_height, target_width = height, width
            print(f"   Using specified target: {target_height}x{target_width}")
        
        # IMPORTANT: Expand masks BEFORE auto-resize to preserve dimensions
        print(f"🎭 Expanding masks (iterations={iterations})...")
        masks = self.expand_masks(masks, iterations)
        print(f"   Expanded masks: {masks.shape}")
        
        # Auto-resize everything for compatibility
        images, masks, final_height, final_width = self.auto_resize_for_compatibility(
            images, masks, target_height, target_width
        )
        
        # Update dimensions for the rest of the pipeline
        height, width = final_height, final_width
        num_frames = input_frames

        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        print(f"✅ Using UniPCMultistepScheduler with {num_inference_steps} steps")

        num_channels_latents = 16
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        print(f"📊 Calculated num_latent_frames: {num_latent_frames} (from {num_frames} frames)")

        # Prepare latents
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            num_latent_frames,
            torch.float16,
            device,
            generator,
            latents,
        )
        print(f"📐 Prepared latents: {latents.shape}")

        # Convert masks and images to device - masks are already expanded and resized
        masks = masks.to(device).half()
        masks[masks > 0] = 1
        
        # Convert images to device
        images = images.to(device).half()
        
        # Masks are now [1, 3, F, H, W] and images are [1, 3, F, H, W] - should be compatible
        print(f"🎭 Final tensor shapes:")
        print(f"   Images: {images.shape}")
        print(f"   Masks: {masks.shape}")

        # Create masked images - both tensors should now have matching shapes
        masked_images = images * (1 - masks)

        # Official VAE latent normalization
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(self.vae.device, torch.float16)
        )

        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(self.vae.device, torch.float16)

        print("🔬 Using official VAE latent normalization")

        # Encode with proper normalization - exactly like official implementation
        with torch.no_grad():
            masked_latents = self.vae.encode(masked_images.half()).latent_dist.mode()
            masks_latents = self.vae.encode((2 * masks - 1.0)).latent_dist.mode()

            # Apply official normalization
            masked_latents = (masked_latents - latents_mean) * latents_std
            masks_latents = (masks_latents - latents_mean) * latents_std

        print(f"🧮 Encoded and normalized latents")
        print(f"   masked_latents: {masked_latents.shape}")
        print(f"   masks_latents: {masks_latents.shape}")
        print(f"   latents: {latents.shape}")

        # CRITICAL: Ensure all latent tensors have matching temporal dimensions
        target_temporal_dim = latents.shape[2]  # Use the prepared latents as reference
        
        # Adjust masked_latents temporal dimension if needed
        if masked_latents.shape[2] != target_temporal_dim:
            print(f"⚠️ Temporal dimension mismatch detected: masked_latents {masked_latents.shape[2]} vs target {target_temporal_dim}")
            if masked_latents.shape[2] > target_temporal_dim:
                # Truncate to match
                masked_latents = masked_latents[:, :, :target_temporal_dim, :, :]
                print(f"   → Truncated masked_latents to {masked_latents.shape}")
            else:
                # Pad by repeating last frame
                pad_frames = target_temporal_dim - masked_latents.shape[2]
                last_frame = masked_latents[:, :, -1:, :, :].repeat(1, 1, pad_frames, 1, 1)
                masked_latents = torch.cat([masked_latents, last_frame], dim=2)
                print(f"   → Padded masked_latents to {masked_latents.shape}")

        # Adjust masks_latents temporal dimension if needed
        if masks_latents.shape[2] != target_temporal_dim:
            print(f"⚠️ Temporal dimension mismatch detected: masks_latents {masks_latents.shape[2]} vs target {target_temporal_dim}")
            if masks_latents.shape[2] > target_temporal_dim:
                # Truncate to match
                masks_latents = masks_latents[:, :, :target_temporal_dim, :, :]
                print(f"   → Truncated masks_latents to {masks_latents.shape}")
            else:
                # Pad by repeating last frame
                pad_frames = target_temporal_dim - masks_latents.shape[2]
                last_frame = masks_latents[:, :, -1:, :, :].repeat(1, 1, pad_frames, 1, 1)
                masks_latents = torch.cat([masks_latents, last_frame], dim=2)
                print(f"   → Padded masks_latents to {masks_latents.shape}")

        # Verify all dimensions match before proceeding
        print(f"✅ Final dimension verification:")
        print(f"   latents: {latents.shape}")
        print(f"   masked_latents: {masked_latents.shape}")
        print(f"   masks_latents: {masks_latents.shape}")
        
        assert latents.shape == masked_latents.shape == masks_latents.shape, \
            f"Dimension mismatch: latents={latents.shape}, masked_latents={masked_latents.shape}, masks_latents={masks_latents.shape}"

        self._num_timesteps = len(timesteps)

        # Official denoising loop - exact copy from official implementation
        print(f"🔥 Starting BMO denoising loop: {len(timesteps)} steps")
        
        for i, t in enumerate(timesteps):
            latent_model_input = latents.to(transformer_dtype)

            # Official concatenation: [latents, masked_latents, masks_latents] = 48 channels
            latent_model_input = torch.cat([latent_model_input, masked_latents, masks_latents], dim=1)
            timestep = t.expand(latents.shape[0])

            if i == 0 or i == len(timesteps) - 1 or i % 10 == 0:
                print(f"   Step {i+1}/{len(timesteps)}: t={t:.1f}, input_shape={latent_model_input.shape}")

            # Model prediction
            noise_pred = self.transformer(
                hidden_states=latent_model_input.half(),
                timestep=timestep
            )[0]

            # Official scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        print(f"🎬 Final latents: {latents.shape} [{latents.min():.3f}, {latents.max():.3f}]")

        # Official denormalization
        latents = latents.half() / latents_std + latents_mean
        print(f"🔄 Denormalized latents: [{latents.min():.3f}, {latents.max():.3f}]")

        # Official VAE decode
        video = self.vae.decode(latents, return_dict=False)[0]
        print(f"📺 Decoded video: {video.shape} [{video.min():.3f}, {video.max():.3f}]")

        # Official post-processing
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        print(f"✅ Final output: {video.shape if hasattr(video, 'shape') else type(video)}")

        return WanPipelineOutput(frames=video) 