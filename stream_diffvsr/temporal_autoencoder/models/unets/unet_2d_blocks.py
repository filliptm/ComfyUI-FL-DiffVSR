# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import torch
import torch.nn as nn
from diffusers.models.activations import get_activation


class TemporalAutoencoderTinyBlock(nn.Module):
    """
    Tiny Autoencoder block used in [`AutoencoderTiny`]. It is a mini residual module consisting of plain conv + ReLU
    blocks.

    Args:
        in_channels (`int`): The number of input channels.
        out_channels (`int`): The number of output channels.
        act_fn (`str`):
            ` The activation function to use. Supported values are `"swish"`, `"mish"`, `"gelu"`, and `"relu"`.

    Returns:
        `torch.Tensor`: A tensor with the same shape as the input tensor, but with the number of channels equal to
        `out_channels`.
    """

    def __init__(self, in_channels: int, out_channels: int, act_fn: str):
        super().__init__()
        act_fn = get_activation(act_fn)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.fuse = nn.ReLU()

        # temporal layers
        self.prev_features = None

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.temporal_processor = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            act_fn,
            nn.Conv1d(out_channels, out_channels, 3, padding=1)
        )

    def forward(self, x):
        current_features = self.conv(x)

        if self.prev_features is not None:
            B, C, H, W = current_features.shape

            pool_kernel = (4, 4)

            avg_pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_kernel)
            current_pooled = avg_pool(current_features)
            prev_pooled = avg_pool(self.prev_features)

            temporal_input = torch.cat([
                current_pooled.view(B, C, -1),
                prev_pooled.view(B, C, -1)
            ], dim=2)

            temporal_out = self.temporal_processor(temporal_input)

            pool_h, pool_w = current_pooled.shape[2], current_pooled.shape[3]
            temporal_out_fuse = self.alpha * temporal_out[:, :, :pool_h * pool_w].view(B, C, pool_h, pool_w) + \
                                (1 - self.alpha) * temporal_out[:, :, -pool_h * pool_w:].view(B, C, pool_h, pool_w)

            temporal_out_fuse = nn.functional.interpolate(
                temporal_out_fuse,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

            current_features = current_features + 0.1 * temporal_out_fuse

        return self.fuse(current_features + self.skip(x))

    def reset_temporal(self):
        self.prev_features = None
