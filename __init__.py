"""
FL DiffVSR - ComfyUI Node Pack for Stream-DiffVSR Video Super-Resolution

A standalone ComfyUI node pack for 4x video upscaling with temporal coherence
using diffusion-based super-resolution.

Based on Stream-DiffVSR: https://arxiv.org/abs/2512.23709
Model: Jamichsu/Stream-DiffVSR (HuggingFace)
"""

from .nodes.model_loader import FL_DiffVSR_LoadModel
from .nodes.upscaler import FL_DiffVSR_Upscale


NODE_CLASS_MAPPINGS = {
    "FL_DiffVSR_LoadModel": FL_DiffVSR_LoadModel,
    "FL_DiffVSR_Upscale": FL_DiffVSR_Upscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_DiffVSR_LoadModel": "FL DiffVSR Load Model",
    "FL_DiffVSR_Upscale": "FL DiffVSR Upscale",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# ASCII art banner
ascii_art = """
⣏⡉ ⡇    ⡏⢱ ⠄ ⣰⡁ ⣰⡁ ⡇⢸ ⢎⡑ ⣏⡱
⠇  ⠧⠤   ⠧⠜ ⠇ ⢸  ⢸  ⠸⠃ ⠢⠜ ⠇⠱
"""
print(f"\033[35m{ascii_art}\033[0m")
print("FL DiffVSR - Video Super-Resolution Node Pack")
