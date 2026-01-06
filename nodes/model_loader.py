"""
FL DiffVSR Model Loader Node
Downloads and loads Stream-DiffVSR model from HuggingFace.
"""

import torch

from ..core.model_manager import get_model_manager
from ..core.pipeline_wrapper import StreamDiffVSRWrapper


class FL_DiffVSR_LoadModel:
    """
    Load Stream-DiffVSR model from HuggingFace.
    Downloads model components to ComfyUI/models/stream_diffvsr/ on first use.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "precision": (["auto", "fp32", "fp16", "bf16"], {"default": "auto"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
            "optional": {
                "enable_xformers": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("FL_DIFFVSR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "FL DiffVSR"

    def load_model(self, precision: str, device: str, enable_xformers: bool = True):
        """Load the Stream-DiffVSR pipeline."""

        # Get model manager
        model_manager = get_model_manager()

        # Download models if needed
        if not model_manager.check_models_exist():
            print("Stream-DiffVSR models not found. Downloading from HuggingFace...")
            model_manager.download_models()

        model_path = model_manager.get_model_path()

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                target_device = torch.device("cuda")
            else:
                target_device = torch.device("cpu")
        else:
            target_device = torch.device(device)

        # Determine dtype
        if precision == "auto":
            if target_device.type == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32
        elif precision == "fp16":
            dtype = torch.float16
        elif precision == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # Only enable xformers on CUDA
        use_xformers = enable_xformers and target_device.type == "cuda"

        # Load pipeline wrapper
        wrapper = StreamDiffVSRWrapper(
            model_path=model_path,
            device=target_device,
            dtype=dtype,
            enable_xformers=use_xformers,
        )

        print(f"Stream-DiffVSR model loaded on {target_device} with {dtype}")

        return (wrapper,)
