# ensemble/base.py
from abc import ABC, abstractmethod
from pathlib import Path
import torch
import torch.nn as nn


class EnsembleModel(ABC, nn.Module):
    """
    Abstract base-class for every ensemble variant.

    Parameters
    ----------
    ckpt_paths : list[str]
        Paths to *.pth files of the base models.
    arch_names : list[str]
        Architecture identifiers (must align with ckpt_paths).
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        ckpt_paths: list[str],
        arch_names: list[str],
        device: str = "cuda",
    ):
        super().__init__()
        self.ckpt_paths = [Path(p) for p in ckpt_paths]
        self.arch_names = arch_names
        self.device = device
        self.experts = self._load_experts()   # list[nn.Module]

    # ------------------------------------------------------------------
    # Methods that every concrete ensemble must implement
    # ------------------------------------------------------------------
    @abstractmethod
    def _load_experts(self) -> list[nn.Module]:
        """Instantiate the base models and load their checkpoints."""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute final probabilities for a batch of images.

        Returns
        -------
        torch.Tensor
            Shape (B, num_classes) – soft probabilities.
        """
        ...