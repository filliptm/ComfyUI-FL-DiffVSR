"""
FL DiffVSR Upscaler Node
Upscales video frames using Stream-DiffVSR with temporal coherence.
"""

import torch
from PIL import Image
import numpy as np

import comfy.utils

from ..core.pipeline_wrapper import StreamDiffVSRWrapper, tensor_to_pil, pil_to_tensor


class FL_DiffVSR_Upscale:
    """
    Upscale video frames using Stream-DiffVSR.
    Processes frames sequentially with temporal coherence.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FL_DIFFVSR_MODEL",),
                "images": ("IMAGE",),
                "inference_steps": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Number of denoising steps (4 recommended for speed/quality balance)"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Classifier-free guidance scale (0 = no guidance)"
                }),
                "chunk_size": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Number of frames to process at once (lower = less VRAM, 0 = process all at once)"
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional text prompt for guidance"
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional negative prompt"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "tooltip": "-1 for random seed"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_images",)
    FUNCTION = "upscale"
    CATEGORY = "FL DiffVSR"

    def upscale(
        self,
        model: StreamDiffVSRWrapper,
        images: torch.Tensor,
        inference_steps: int,
        guidance_scale: float,
        chunk_size: int = 8,
        prompt: str = "",
        negative_prompt: str = "",
        seed: int = -1,
    ):
        """
        Upscale images using Stream-DiffVSR pipeline.

        Args:
            model: StreamDiffVSRWrapper instance
            images: Tensor of shape [B, H, W, C] in ComfyUI format (0-1 range)
            inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            chunk_size: Number of frames per chunk (lower = less VRAM)
            prompt: Text prompt for guidance
            negative_prompt: Negative prompt
            seed: Random seed (-1 for random)

        Returns:
            Upscaled images tensor [B, H*4, W*4, C]
        """
        num_frames = images.shape[0]
        total_steps = num_frames * inference_steps
        print(f"FL DiffVSR: Processing {num_frames} frames with {inference_steps} steps ({total_steps} total steps)...")
        if chunk_size > 0:
            print(f"FL DiffVSR: Using chunk_size={chunk_size} for VRAM-efficient processing")

        # Set seed if specified
        generator = None
        if seed != -1:
            generator = torch.Generator(device=model.device).manual_seed(seed)

        # Convert ComfyUI tensors to PIL images
        pil_images = []
        for i in range(num_frames):
            frame = images[i]  # [H, W, C]
            pil_img = tensor_to_pil(frame)
            pil_images.append(pil_img)

        # Setup ComfyUI progress bar
        pbar = comfy.utils.ProgressBar(total_steps)
        current_progress = [0]  # Use list to allow modification in closure

        def progress_callback(frame_idx, step_idx, total_frames, total_steps_per_frame, latents):
            # Calculate overall progress
            overall_step = frame_idx * total_steps_per_frame + step_idx + 1
            steps_to_update = overall_step - current_progress[0]
            if steps_to_update > 0:
                pbar.update(steps_to_update)
                current_progress[0] = overall_step

        # Process through pipeline with chunking support
        upscaled_pil = model.process_frames(
            images=pil_images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            progress_callback=progress_callback,
            chunk_size=chunk_size,
        )

        # Ensure progress bar completes
        remaining = total_steps - current_progress[0]
        if remaining > 0:
            pbar.update(remaining)

        # Convert back to ComfyUI tensors
        upscaled_tensors = []
        for pil_img in upscaled_pil:
            tensor = pil_to_tensor(pil_img)  # [1, H, W, C]
            upscaled_tensors.append(tensor)

        # Concatenate all frames [B, H*4, W*4, C]
        output = torch.cat(upscaled_tensors, dim=0)

        print(f"FL DiffVSR: Upscaling complete. Output shape: {list(output.shape)}")

        return (output,)
