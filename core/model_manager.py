"""
Model Manager for FL DiffVSR
Handles downloading and managing Stream-DiffVSR models from HuggingFace.
"""

import os
from pathlib import Path
from typing import Optional

import folder_paths


class ModelManager:
    """Manages Stream-DiffVSR model downloading and paths."""

    HF_REPO_ID = "Jamichsu/Stream-DiffVSR"
    MODEL_FOLDER_NAME = "stream_diffvsr"

    def __init__(self):
        self.models_dir = self._get_models_dir()

    def _get_models_dir(self) -> str:
        """Get the path to the stream_diffvsr models directory."""
        models_base = folder_paths.models_dir
        models_dir = os.path.join(models_base, self.MODEL_FOLDER_NAME)
        os.makedirs(models_dir, exist_ok=True)
        return models_dir

    def get_model_path(self) -> str:
        """Get the path where models are/will be stored."""
        return self.models_dir

    def check_models_exist(self) -> bool:
        """Check if all required model files exist."""
        required_dirs = [
            "unet",
            "controlnet",
            "vae",
            "text_encoder",
            "tokenizer",
            "scheduler",
        ]

        for dir_name in required_dirs:
            dir_path = os.path.join(self.models_dir, dir_name)
            if not os.path.exists(dir_path):
                return False

            # Check for at least one file in each directory
            if not any(os.scandir(dir_path)):
                return False

        return True

    def download_models(self, force: bool = False) -> str:
        """
        Download all model components from HuggingFace.

        Args:
            force: If True, download even if models already exist

        Returns:
            Path to the downloaded models
        """
        if self.check_models_exist() and not force:
            print(f"Stream-DiffVSR models already exist at {self.models_dir}")
            return self.models_dir

        try:
            from huggingface_hub import snapshot_download

            print(f"Downloading Stream-DiffVSR models from {self.HF_REPO_ID}...")
            print(f"Target directory: {self.models_dir}")

            local_dir = snapshot_download(
                repo_id=self.HF_REPO_ID,
                local_dir=self.models_dir,
                local_dir_use_symlinks=False,
                ignore_patterns=[
                    "*.md",
                    "*.txt",
                    ".git*",
                    "*.py",
                    "*.yml",
                    "*.yaml",
                ],
            )

            print(f"Models downloaded successfully to: {local_dir}")
            return local_dir

        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download models. "
                "Please install it with: pip install huggingface_hub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download Stream-DiffVSR models: {e}")

    def get_component_path(self, component: str) -> str:
        """Get the path to a specific model component."""
        return os.path.join(self.models_dir, component)


# Singleton instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the singleton ModelManager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
