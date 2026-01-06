"""
Pipeline Wrapper for FL DiffVSR
Wraps the Stream-DiffVSR pipeline for ComfyUI integration.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Optional, List, Union

from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


class StreamDiffVSRWrapper:
    """
    Wrapper around StreamDiffVSRPipeline for ComfyUI integration.
    Handles model loading, optical flow, and frame-by-frame processing.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype,
        enable_xformers: bool = True,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype

        # Load pipeline components
        self._load_pipeline(enable_xformers)

        # Load optical flow model (RAFT)
        self._load_flow_model()

        # Temporal state
        self.prev_frame_rgb = None
        self.frame_count = 0

    def _load_pipeline(self, enable_xformers: bool):
        """Load the Stream-DiffVSR pipeline."""
        from diffusers import UNet2DConditionModel, ControlNetModel
        from transformers import CLIPTextModel, CLIPTokenizer

        # Import local modules
        from ..stream_diffvsr.temporal_autoencoder.autoencoder_tiny import TemporalAutoencoderTiny
        from ..stream_diffvsr.pipeline.stream_diffvsr_pipeline import StreamDiffVSRPipeline
        from ..stream_diffvsr.scheduler.ddim_scheduler import DDIMScheduler

        print("Loading Stream-DiffVSR pipeline components...")

        # Load UNet
        print("  Loading UNet...")
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_path, subfolder="unet", torch_dtype=self.dtype
        ).to(self.device)

        # Load ControlNet
        print("  Loading ControlNet...")
        self.controlnet = ControlNetModel.from_pretrained(
            self.model_path, subfolder="controlnet", torch_dtype=self.dtype
        ).to(self.device)

        # Load Temporal VAE
        print("  Loading Temporal VAE...")
        self.vae = TemporalAutoencoderTiny.from_pretrained(
            self.model_path, subfolder="vae", torch_dtype=self.dtype
        ).to(self.device)

        # Load text encoder and tokenizer
        print("  Loading Text Encoder...")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_path, subfolder="text_encoder", torch_dtype=self.dtype
        ).to(self.device)

        # Load tokenizer - the Stream-DiffVSR model's tokenizer is incomplete (missing merges.txt)
        # So we load from openai/clip-vit-large-patch14 which is the base CLIP model
        print("  Loading Tokenizer...")
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.model_path, subfolder="tokenizer"
            )
        except (TypeError, OSError) as e:
            print(f"  Local tokenizer incomplete, loading from openai/clip-vit-large-patch14...")
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )

        # Load scheduler
        print("  Loading Scheduler...")
        self.scheduler = DDIMScheduler.from_pretrained(
            self.model_path, subfolder="scheduler"
        )

        # Create pipeline
        self.pipeline = StreamDiffVSRPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )

        # Enable memory optimizations
        if enable_xformers:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                print("  xformers memory efficient attention enabled")
            except Exception as e:
                print(f"  Could not enable xformers: {e}")

        print("Stream-DiffVSR pipeline loaded successfully!")

    def _load_flow_model(self):
        """Load RAFT optical flow model."""
        print("Loading RAFT optical flow model...")
        self.flow_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        self.flow_model = self.flow_model.to(self.device).eval()
        self.flow_model.requires_grad_(False)
        print("RAFT model loaded successfully!")

    def reset_temporal_state(self):
        """Reset temporal state for new video sequence."""
        self.prev_frame_rgb = None
        self.frame_count = 0
        self.vae.reset_temporal_condition()

    def process_frames_chunked(
        self,
        images: List[Image.Image],
        chunk_size: int = 8,
        prompt: str = "",
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        generator: Optional[torch.Generator] = None,
        progress_callback: Optional[callable] = None,
    ) -> List[Image.Image]:
        """
        Process frames in memory-efficient chunks.

        Args:
            images: List of PIL Images to upscale
            chunk_size: Number of frames per chunk (lower = less VRAM)
            prompt: Text prompt for guidance
            negative_prompt: Negative text prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            generator: Random generator for reproducibility
            progress_callback: Optional callback for progress updates

        Returns:
            List of upscaled PIL Images
        """
        all_results = []
        prev_frame_rgb = None
        prev_upscaled_for_flow = None
        total_frames = len(images)

        # Calculate number of chunks
        num_chunks = (total_frames + chunk_size - 1) // chunk_size

        print(f"  Processing {total_frames} frames in {num_chunks} chunk(s) of up to {chunk_size} frames each")

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_frames)
            chunk_images = images[start_idx:end_idx]

            print(f"  Processing chunk {chunk_idx + 1}/{num_chunks} (frames {start_idx}-{end_idx - 1})")

            # Reset temporal state but pass previous frame context
            self.vae.reset_temporal_condition()

            # Process chunk with previous frame state for continuity
            output, prev_frame_rgb, prev_upscaled_for_flow = self.pipeline(
                prompt=prompt,
                images=chunk_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt if negative_prompt else None,
                generator=generator,
                of_model=self.flow_model,
                of_rescale_factor=1,
                output_type="pil",
                callback=progress_callback,
                callback_steps=1,
                # Pass previous frame state for chunk continuity
                prev_frame_rgb=prev_frame_rgb,
                prev_upscaled_for_flow=prev_upscaled_for_flow,
                # Pass frame offset for progress callback
                frame_offset=start_idx,
            )

            # Extract results from this chunk
            for img_list in output.images:
                if isinstance(img_list, list):
                    all_results.extend(img_list)
                else:
                    all_results.append(img_list)

            # Clear VRAM between chunks
            torch.cuda.empty_cache()

        return all_results

    def process_frames(
        self,
        images: List[Image.Image],
        prompt: str = "",
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        generator: Optional[torch.Generator] = None,
        progress_callback: Optional[callable] = None,
        chunk_size: int = 0,
    ) -> List[Image.Image]:
        """
        Process a list of frames through the pipeline.

        Args:
            images: List of PIL Images to upscale
            prompt: Text prompt for guidance
            negative_prompt: Negative text prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            generator: Random generator for reproducibility
            progress_callback: Optional callback for progress updates
            chunk_size: Number of frames per chunk (0 = process all at once)

        Returns:
            List of upscaled PIL Images
        """
        # Use chunked processing if chunk_size > 0 and we have more frames than chunk_size
        if chunk_size > 0 and len(images) > chunk_size:
            return self.process_frames_chunked(
                images=images,
                chunk_size=chunk_size,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                progress_callback=progress_callback,
            )

        # Reset temporal state for new sequence
        self.reset_temporal_state()

        # Run pipeline (original behavior for small batches or chunk_size=0)
        output, _, _ = self.pipeline(
            prompt=prompt,
            images=images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt if negative_prompt else None,
            generator=generator,
            of_model=self.flow_model,
            of_rescale_factor=1,
            output_type="pil",
            callback=progress_callback,
            callback_steps=1,
            prev_frame_rgb=None,
            prev_upscaled_for_flow=None,
            frame_offset=0,
        )

        # Extract results
        results = []
        for img_list in output.images:
            if isinstance(img_list, list):
                results.extend(img_list)
            else:
                results.append(img_list)

        return results

    def to(self, device: torch.device):
        """Move all models to specified device."""
        self.device = device
        self.unet = self.unet.to(device)
        self.controlnet = self.controlnet.to(device)
        self.vae = self.vae.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.flow_model = self.flow_model.to(device)
        return self


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a ComfyUI tensor [H, W, C] (0-1 range) to PIL Image.
    """
    arr = tensor.cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode='RGB')


def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to ComfyUI tensor [1, H, W, C] (0-1 range).
    """
    arr = np.array(pil_image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr)
    return tensor.unsqueeze(0)  # [H, W, C] -> [1, H, W, C]
