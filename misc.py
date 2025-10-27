"""
Author: Sagar Chakraborty
Updated: Modernized utilities (legacy file - most functionality moved to train.py)

Note: This file is kept for backward compatibility but is no longer used
in the modern implementation. The new train.py uses PyTorch DataLoader
directly instead of torchtext.legacy and pytorch-ignite.
"""

import glob
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Utility class for loading PyTorch model checkpoints.

    This is a legacy class kept for backward compatibility.
    For new code, use torch.load() directly.
    """

    def __init__(
        self,
        model: nn.Module,
        dirname: str,
        filename_prefix: str
    ):
        """
        Initialize ModelLoader.

        Args:
            model: PyTorch model to load weights into
            dirname: Directory containing model checkpoints
            filename_prefix: Prefix of checkpoint files
        """
        self._dirname = Path(dirname)
        self._fname_prefix = filename_prefix
        self._model = model
        self._fname = self._dirname / filename_prefix
        self.skip_load = False

        # Validate model
        if not isinstance(model, nn.Module):
            raise ValueError("model should be an instance of nn.Module")

        # Check if directory exists
        if not self._dirname.exists():
            self.skip_load = True
            logger.warning(
                f"Directory '{self._dirname}' not found, skipping model restoration"
            )
            return

        # Check if any checkpoint files exist
        checkpoint_pattern = str(self._fname) + "_*"
        if not list(Path(dirname).glob(filename_prefix + "_*")):
            self.skip_load = True
            logger.warning(
                f"No checkpoint files found matching '{checkpoint_pattern}', "
                "skipping model restoration"
            )

    def _load(self, path_pattern: str) -> None:
        """
        Load model weights from checkpoint.

        Args:
            path_pattern: Glob pattern for checkpoint files
        """
        if self.skip_load:
            return

        # Find all matching checkpoints
        models = sorted(glob.glob(path_pattern))

        if not models:
            logger.warning(f"No models found matching pattern: {path_pattern}")
            return

        # Load the latest checkpoint
        latest_model = models[-1]

        try:
            # Load state dict
            state_dict = torch.load(latest_model, map_location='cpu')

            # Handle DataParallel models
            if isinstance(self._model, nn.parallel.DataParallel):
                self._model.module.load_state_dict(state_dict)
            else:
                self._model.load_state_dict(state_dict)

            logger.info(f"Successfully loaded model from: {latest_model}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def __call__(self, infix_name: str) -> None:
        """
        Load model with specific infix in filename.

        Args:
            infix_name: Infix to use in checkpoint filename pattern
        """
        path_pattern = str(self._fname) + "_" + infix_name + "_*"
        self._load(path_pattern)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Modern utility function to load model checkpoint.

    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load model on (default: CPU)

    Returns:
        Model with loaded weights
    """
    if device is None:
        device = torch.device('cpu')

    checkpoint_file = Path(checkpoint_path)

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)

    logger.info("Checkpoint loaded successfully")

    return model


def save_checkpoint(
    model: nn.Module,
    save_path: str,
    additional_info: Optional[Dict] = None
) -> None:
    """
    Save model checkpoint with optional additional information.

    Args:
        model: PyTorch model to save
        save_path: Path to save checkpoint
        additional_info: Optional dictionary with additional info (epoch, optimizer state, etc.)
    """
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    # Prepare checkpoint
    if additional_info is not None:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            **additional_info
        }
    else:
        checkpoint = model.state_dict()

    # Save checkpoint
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to: {save_path}")


# Legacy function kept for backward compatibility
def create_supervised_evaluator(model, inference_fn, metrics=None, cuda=False):
    """
    Legacy function kept for backward compatibility.

    For new code, use the modern training loop in train.py instead.
    This function was used with pytorch-ignite but is no longer needed.
    """
    logger.warning(
        "create_supervised_evaluator is a legacy function. "
        "Please use the modern SentimentTrainer class in train.py instead."
    )
    raise DeprecationWarning(
        "This function is deprecated. Use the modern training pipeline in train.py"
    )


if __name__ == "__main__":
    print("This is a utility module. Import it in your training script.")
