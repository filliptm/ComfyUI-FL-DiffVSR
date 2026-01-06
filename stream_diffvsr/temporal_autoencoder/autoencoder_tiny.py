# Copyright 2024 Ollin Boer Bohan and The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.modeling_utils import ModelMixin
from .vae import DecoderOutput, TemporalDecoderTiny, EncoderTiny
from .models.unets.unet_2d_blocks import TemporalAutoencoderTinyBlock


@dataclass
class TemporalAutoencoderTinyOutput(BaseOutput):
    """
    Output of TemporalAutoencoderTiny encoding method.

    Args:
        latents (`torch.Tensor`): Encoded outputs of the `Encoder`.
    """

    latents: torch.Tensor


class TemporalAutoencoderTiny(ModelMixin, ConfigMixin):
    """
    A tiny distilled VAE model for encoding images into latents and decoding latent representations into images.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        encoder_block_out_channels: Tuple[int, ...] = (64, 64, 64, 64),
        decoder_block_out_channels: Tuple[int, ...] = (64, 64, 64, 64),
        act_fn: str = "relu",
        upsample_fn: str = "nearest",
        latent_channels: int = 4,
        upsampling_scaling_factor: int = 2,
        num_encoder_blocks: Tuple[int, ...] = (1, 3, 3, 3),
        num_decoder_blocks: Tuple[int, ...] = (3, 3, 3, 1),
        latent_magnitude: int = 3,
        latent_shift: float = 0.5,
        force_upcast: bool = False,
        scaling_factor: float = 1.0,
        shift_factor: float = 0.0,
        block_out_channels: Tuple[int, ...] = None,  # For compatibility with saved configs
    ):
        super().__init__()

        if len(encoder_block_out_channels) != len(num_encoder_blocks):
            raise ValueError("`encoder_block_out_channels` should have the same length as `num_encoder_blocks`.")
        if len(decoder_block_out_channels) != len(num_decoder_blocks):
            raise ValueError("`decoder_block_out_channels` should have the same length as `num_decoder_blocks`.")

        self.encoder = EncoderTiny(
            in_channels=in_channels,
            out_channels=latent_channels,
            num_blocks=num_encoder_blocks,
            block_out_channels=encoder_block_out_channels,
            act_fn=act_fn,
        )

        self.encoder.requires_grad_(False)

        self.decoder = TemporalDecoderTiny(
            in_channels=latent_channels,
            out_channels=out_channels,
            num_blocks=num_decoder_blocks,
            block_out_channels=decoder_block_out_channels,
            upsampling_scaling_factor=upsampling_scaling_factor,
            act_fn=act_fn,
            upsample_fn=upsample_fn,
        )

        self.decoder.requires_grad_(False)

        for name, param in self.decoder.named_parameters():
            if "alpha" in name or "temporal_processor" in name:
                param.requires_grad_(True)

        self.latent_magnitude = latent_magnitude
        self.latent_shift = latent_shift
        self.scaling_factor = scaling_factor

        self.use_slicing = False
        self.use_tiling = False

        self.spatial_scale_factor = 2**out_channels
        self.tile_overlap_factor = 0.125
        self.tile_sample_min_size = 512
        self.tile_latent_min_size = self.tile_sample_min_size // self.spatial_scale_factor

        self.register_to_config(block_out_channels=decoder_block_out_channels)
        self.register_to_config(force_upcast=False)

    def reset_temporal_condition(self):
        """reset temporal memory"""
        for module in self.encoder.layers:
            if isinstance(module, TemporalAutoencoderTinyBlock):
                module.reset_temporal()
        for module in self.decoder.layers:
            if isinstance(module, TemporalAutoencoderTinyBlock):
                module.reset_temporal()

    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        if isinstance(module, (EncoderTiny, TemporalDecoderTiny)):
            module.gradient_checkpointing = value

    def scale_latents(self, x: torch.Tensor) -> torch.Tensor:
        """raw latents -> [0, 1]"""
        return x.div(2 * self.latent_magnitude).add(self.latent_shift).clamp(0, 1)

    def unscale_latents(self, x: torch.Tensor) -> torch.Tensor:
        """[0, 1] -> raw latents"""
        return x.sub(self.latent_shift).mul(2 * self.latent_magnitude)

    def enable_slicing(self) -> None:
        self.use_slicing = True

    def disable_slicing(self) -> None:
        self.use_slicing = False

    def enable_tiling(self, use_tiling: bool = True) -> None:
        self.use_tiling = use_tiling

    def disable_tiling(self) -> None:
        self.enable_tiling(False)

    def _tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        sf = self.spatial_scale_factor
        tile_size = self.tile_sample_min_size
        blend_size = int(tile_size * self.tile_overlap_factor)
        traverse_size = tile_size - blend_size

        ti = range(0, x.shape[-2], traverse_size)
        tj = range(0, x.shape[-1], traverse_size)

        blend_masks = torch.stack(
            torch.meshgrid([torch.arange(tile_size / sf) / (blend_size / sf - 1)] * 2, indexing="ij")
        )
        blend_masks = blend_masks.clamp(0, 1).to(x.device)

        out = torch.zeros(x.shape[0], 4, x.shape[-2] // sf, x.shape[-1] // sf, device=x.device)
        for i in ti:
            for j in tj:
                tile_in = x[..., i : i + tile_size, j : j + tile_size]
                tile_out = out[..., i // sf : (i + tile_size) // sf, j // sf : (j + tile_size) // sf]
                tile = self.encoder(tile_in)
                h, w = tile.shape[-2], tile.shape[-1]
                blend_mask_i = torch.ones_like(blend_masks[0]) if i == 0 else blend_masks[0]
                blend_mask_j = torch.ones_like(blend_masks[1]) if j == 0 else blend_masks[1]
                blend_mask = blend_mask_i * blend_mask_j
                tile, blend_mask = tile[..., :h, :w], blend_mask[..., :h, :w]
                tile_out.copy_(blend_mask * tile + (1 - blend_mask) * tile_out)
        return out

    def _tiled_decode(self, x: torch.Tensor) -> torch.Tensor:
        sf = self.spatial_scale_factor
        tile_size = self.tile_latent_min_size
        blend_size = int(tile_size * self.tile_overlap_factor)
        traverse_size = tile_size - blend_size

        ti = range(0, x.shape[-2], traverse_size)
        tj = range(0, x.shape[-1], traverse_size)

        blend_masks = torch.stack(
            torch.meshgrid([torch.arange(tile_size * sf) / (blend_size * sf - 1)] * 2, indexing="ij")
        )
        blend_masks = blend_masks.clamp(0, 1).to(x.device)

        out = torch.zeros(x.shape[0], 3, x.shape[-2] * sf, x.shape[-1] * sf, device=x.device)
        for i in ti:
            for j in tj:
                tile_in = x[..., i : i + tile_size, j : j + tile_size]
                tile_out = out[..., i * sf : (i + tile_size) * sf, j * sf : (j + tile_size) * sf]
                tile = self.decoder(tile_in)
                h, w = tile.shape[-2], tile.shape[-1]
                blend_mask_i = torch.ones_like(blend_masks[0]) if i == 0 else blend_masks[0]
                blend_mask_j = torch.ones_like(blend_masks[1]) if j == 0 else blend_masks[1]
                blend_mask = (blend_mask_i * blend_mask_j)[..., :h, :w]
                tile_out.copy_(blend_mask * tile + (1 - blend_mask) * tile_out)
        return out

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True, return_layers_features: bool = True, return_features_only: bool = False) -> Union[TemporalAutoencoderTinyOutput, Tuple[torch.Tensor]]:
        layer_features = [] if return_layers_features else None

        if self.use_slicing and x.shape[0] > 1:
            output = [
                self._tiled_encode(x_slice) if self.use_tiling else self.encoder(x_slice)
                for x_slice in x.split(1)
            ]
            output = torch.cat(output)
        else:
            if self.use_tiling:
                output = self._tiled_encode(x)
            elif return_layers_features:
                current_features = x
                for module in self.encoder.layers:
                    current_features = module(current_features)

                    if isinstance(module, TemporalAutoencoderTinyBlock):
                        layer_features.append(current_features)

                if return_features_only:
                    return layer_features

            output = self.encoder(x)

        if not return_dict:
            return (output,), layer_features

        return TemporalAutoencoderTinyOutput(latents=output)

    @apply_forward_hook
    def decode(
        self, x: torch.Tensor, temporal_features=None, generator: Optional[torch.Generator] = None, return_dict: bool = True
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        if self.use_slicing and x.shape[0] > 1:
            output = [
                self._tiled_decode(x_slice) if self.use_tiling else self.decoder(x_slice) for x_slice in x.split(1)
            ]
            output = torch.cat(output)
        elif temporal_features is not None:
            block_idx = 0
            for module in self.decoder.layers:
                if isinstance(module, TemporalAutoencoderTinyBlock):
                    module.prev_features = temporal_features[block_idx]
                    block_idx += 1
            output = self.decoder(x)
        else:
            output = self._tiled_decode(x) if self.use_tiling else self.decoder(x)

        if not return_dict:
            return (output,)

        return DecoderOutput(sample=output)

    def forward(
        self,
        sample: torch.Tensor,
        previous_sample: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        layer_features = None

        if previous_sample is not None:
            prev_enc, layer_features = self.encode(previous_sample, return_dict=return_dict)

        if layer_features is not None:
            temporal_features = layer_features[::-1]
        else:
            temporal_features = None

        dec = self.decode(sample, temporal_features=temporal_features, return_dict=return_dict)[0]

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)
