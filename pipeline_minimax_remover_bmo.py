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
        """Official resize method - exact copy from official code"""
        bsz, _, _, _, _ = images.shape
        images = rearrange(images, "b c f w h -> (b f) c w h")
        images = functional.interpolate(images, (w, h), mode='bilinear')
        images = rearrange(images, "(b f) c w h -> b c f w h", b=bsz)
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
        BMO inference method - simplified and clean
        """
        print("🔧 Using BMO MiniMax-Remover Pipeline")
        
        self._current_timestep = None
        self._interrupt = False
        device = self._execution_device
        batch_size = 1
        transformer_dtype = torch.float16

        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        print(f"✅ Using UniPCMultistepScheduler with {num_inference_steps} steps")

        num_channels_latents = 16
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # Prepare latents - use exact official approach
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

        # Process masks exactly like official implementation
        masks = self.expand_masks(masks, iterations)
        masks = self.resize(masks, height, width).to(device).half()
        masks[masks > 0] = 1
        print(f"🎭 Processed masks: {masks.shape}")

        # Process images exactly like official implementation
        images = rearrange(images, "f h w c -> c f h w")
        images = self.resize(images[None, ...], height, width).to(device).half()
        print(f"🖼️ Processed images: {images.shape}")

        # Create masked images
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
            masks_latents = self.vae.encode(2 * masks.half() - 1.0).latent_dist.mode()

            # Apply official normalization
            masked_latents = (masked_latents - latents_mean) * latents_std
            masks_latents = (masks_latents - latents_mean) * latents_std

        print(f"🧮 Encoded and normalized latents")
        print(f"   masked_latents: {masked_latents.shape}")
        print(f"   masks_latents: {masks_latents.shape}")
        print(f"   latents: {latents.shape}")

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