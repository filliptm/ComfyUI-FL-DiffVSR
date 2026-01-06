# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor, is_compiled_module
from diffusers.pipelines import DiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker

from ..util import flow_utils as of
from ..temporal_autoencoder.autoencoder_tiny import TemporalAutoencoderTiny
from ..scheduler.ddim_scheduler import DDIMScheduler


logger = logging.get_logger(__name__)


class StreamDiffVSRPipeline(
    DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin
):
    """
    Stream-DiffVSR Pipeline for video super-resolution.

    Args:
        vae: Temporal VAE model for encoding/decoding
        text_encoder: CLIP text encoder
        tokenizer: CLIP tokenizer
        unet: UNet2DConditionModel for denoising
        controlnet: ControlNet for temporal conditioning
        scheduler: DDIM scheduler
        safety_checker: Optional safety checker
        feature_extractor: Optional feature extractor
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: Union[AutoencoderKL, TemporalAutoencoderTiny],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        scheduler: Union[KarrasDiffusionSchedulers, DDIMScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety checker."
            )

        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=True
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        self.vae.disable_tiling()

    def enable_model_cpu_offload(self, gpu_id=0):
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        if self.safety_checker is not None:
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        cpu_offload_with_hook(self.controlnet, device)
        self.final_offload_hook = hook

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt

            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def compute_flows(self, of_model, images, rescale_factor=1):
        print('Computing forward flows...')
        forward_flows = []
        for i in range(1, len(images)):
            # RAFT optical flow model requires float32 input
            prev_image = images[i - 1].float()
            cur_image = images[i].float()
            fflow = of.get_flow(of_model, cur_image, prev_image, rescale_factor=rescale_factor)
            forward_flows.append(fflow)
        return forward_flows

    def compute_single_flow(self, of_model, prev_image, cur_image, rescale_factor=1):
        """Compute optical flow between two adjacent frames.

        To save VRAM, we compute flow at 1/4 resolution and upscale it back.
        This is much more memory efficient for high-resolution inputs.
        """
        # RAFT optical flow model requires float32 input
        prev_image_f32 = prev_image.float()
        cur_image_f32 = cur_image.float()

        # Downscale images for RAFT to save VRAM (RAFT is very memory hungry at high res)
        # Use 1/2 resolution for flow computation (better quality than 1/4)
        flow_scale = 2
        _, _, h, w = prev_image_f32.shape
        small_h, small_w = h // flow_scale, w // flow_scale

        prev_small = F.interpolate(prev_image_f32, size=(small_h, small_w), mode='bilinear', align_corners=False)
        cur_small = F.interpolate(cur_image_f32, size=(small_h, small_w), mode='bilinear', align_corners=False)

        # Compute flow at lower resolution
        fflow_small = of.get_flow(of_model, cur_small, prev_small, rescale_factor=rescale_factor)

        # Upscale flow back to original resolution and scale the flow values
        # Flow is in [B, H, W, 2] format after get_flow
        fflow_small_permuted = fflow_small.permute(0, 3, 1, 2)  # [B, 2, H, W]
        fflow_upscaled = F.interpolate(fflow_small_permuted, size=(h, w), mode='bilinear', align_corners=False)
        fflow = fflow_upscaled.permute(0, 2, 3, 1)  # [B, H, W, 2]

        # Scale flow values to match the resolution change
        fflow = fflow * flow_scale

        return fflow

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        images: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        of_model=None,
        of_rescale_factor: int = 1,
        timesteps_to_be_used: Optional[List[float]] = None,
        # New parameters for chunked processing
        prev_frame_rgb: Optional[torch.FloatTensor] = None,
        prev_upscaled_for_flow: Optional[torch.FloatTensor] = None,
        frame_offset: int = 0,
    ):
        """
        Run the Stream-DiffVSR pipeline.
        """
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [control_guidance_end]

        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # Get model dtype early for consistency
        model_dtype = self.unet.dtype

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # Ensure prompt embeds are in model dtype
        prompt_embeds = prompt_embeds.to(dtype=model_dtype)

        # Prepare timesteps
        if timesteps_to_be_used is None:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
        else:
            self.scheduler.set_timesteps(timesteps=timesteps_to_be_used, device=device)
        timesteps = self.scheduler.timesteps

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        interp_mode = 'bilinear' if of_rescale_factor == 1 else 'nearest'

        # Initialize state from previous chunk if provided
        rgb_for_warpping_to_next_frame = prev_frame_rgb
        prev_upscaled = prev_upscaled_for_flow


        # Store raw images for per-frame processing
        raw_images = images
        output_images = []
        num_channels_latents = self.vae.config.latent_channels

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        total_frames = len(raw_images)

        with self.progress_bar(total=len(timesteps)*total_frames) as progress_bar:
            for num_image, raw_image in enumerate(raw_images):
                # === PROCESS ONE FRAME AT A TIME (VRAM efficient) ===

                # Preprocess current frame only - DO NOT pass height/width to avoid resizing
                # The preprocessor should just normalize, not resize the input
                image = self.control_image_processor.preprocess(raw_image).to(dtype=model_dtype, device=device)

                # Upscale current frame only (4x bicubic for flow/conditioning)
                upscaled = F.interpolate(image, scale_factor=4, mode='bicubic').to(dtype=model_dtype)

                # Get dimensions from upscaled output (for latent preparation)
                frame_height, frame_width = upscaled.shape[-2:]

                # Prepare latent for current frame only
                latent = self.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    frame_height,
                    frame_width,
                    model_dtype,
                    device,
                    generator
                )

                # Compute flow only between adjacent frames
                flow = None
                if prev_upscaled is not None:
                    flow = self.compute_single_flow(of_model, prev_upscaled, upscaled, rescale_factor=of_rescale_factor)

                dec_temporal_features = None
                warped_prev_est = None

                # Compute Temporal Texture Guidance if we have previous frame
                if rgb_for_warpping_to_next_frame is not None and flow is not None:
                    warped_prev_est = of.flow_warp(rgb_for_warpping_to_next_frame, flow, interp_mode=interp_mode)
                    warped_prev_est = warped_prev_est.to(dtype=model_dtype)
                    enc_layer_features = self.vae.encode(warped_prev_est, return_features_only=True)
                    dec_temporal_features = enc_layer_features[::-1]

                for i, t in enumerate(timesteps):
                    # Ensure timestep is on the correct device
                    t = t.to(device)

                    latent_model_input = torch.cat([latent] * 2) if do_classifier_free_guidance else latent
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # Double image for CFG (unconditional + conditional)
                    image_input = torch.cat([image] * 2) if do_classifier_free_guidance else image
                    latent_model_input = torch.cat([latent_model_input, image_input], dim=1)

                    # controlnet(s) inference
                    if guess_mode and do_classifier_free_guidance:
                        control_model_input = latent
                        control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                        controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                    else:
                        control_model_input = latent_model_input
                        controlnet_prompt_embeds = prompt_embeds

                    if isinstance(controlnet_keep[i], list):
                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[i]

                    # Use ControlNet only if we have warped previous estimate
                    if warped_prev_est is None:
                        down_block_res_samples = None
                        mid_block_res_sample = None
                    else:
                        # Double warped_prev_est for CFG
                        controlnet_cond_input = torch.cat([warped_prev_est] * 2) if do_classifier_free_guidance else warped_prev_est
                        down_block_res_samples, mid_block_res_sample = self.controlnet(
                            control_model_input,
                            t,
                            encoder_hidden_states=controlnet_prompt_embeds,
                            controlnet_cond=controlnet_cond_input,
                            conditioning_scale=cond_scale,
                            guess_mode=guess_mode,
                            return_dict=False,
                            timestep_cond=None
                        )
                        if guess_mode and do_classifier_free_guidance:
                            down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                            mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    step_output = self.scheduler.step(noise_pred, t, latent, **extra_step_kwargs)
                    latent, x0_est = step_output.prev_sample, step_output.pred_original_sample

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            # Pass frame index (with offset), step index, total frames, total steps
                            callback(num_image + frame_offset, i, total_frames, len(timesteps), latent)

                if not output_type == "latent":
                    decoded_image = self.vae.decode(latent / self.vae.config.scaling_factor, temporal_features=dec_temporal_features, return_dict=False)[0]
                else:
                    decoded_image = latent

                # Update state for next frame
                # Clone tensors to avoid reference issues between frames
                rgb_for_warpping_to_next_frame = decoded_image.clone()
                prev_upscaled = upscaled.clone()

                has_nsfw_concept = None
                do_denormalize = [True] * decoded_image[0].shape[0]
                final_image = self.image_processor.postprocess(decoded_image, output_type=output_type, do_denormalize=do_denormalize)
                output_images.append(final_image)

                self.vae.reset_temporal_condition()

                # Free memory for this frame
                del image, latent, upscaled, decoded_image
                if flow is not None:
                    del flow
                if warped_prev_est is not None:
                    del warped_prev_est

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        # Return output along with state for next chunk
        output = StableDiffusionPipelineOutput(images=output_images, nsfw_content_detected=None)
        return output, rgb_for_warpping_to_next_frame, prev_upscaled
