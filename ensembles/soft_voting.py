# ensemble/soft_voting.py
"""
Soft‐Voting Ensemble
~~~~~~~~~~~~~~~~~~~~
Combines N expert models by averaging their raw logits and
returning calibrated probabilities (softmax).
"""

from pathlib import Path
import torch
import torch.nn.functional as F

from .base import EnsembleModel
# --- import the model builders you actually use -------------------
from models.resnet_model import get_resnet_model
from models.vgg19_model  import get_vgg19_model
from models.vit_model    import get_vit_model
# add further imports (e.g. get_dinov2_model) as needed
# ------------------------------------------------------------------

_MODEL_MAP = {
    "resnet50": get_resnet_model,
    "vgg19":    get_vgg19_model,
    "vit":      get_vit_model,
    # "dinov2":  get_dinov2_model,
}


class SoftVotingEnsemble(EnsembleModel):
    """Average raw logits from each expert model."""

    # --------------------------------------------------------------
    # required abstract methods from EnsembleModel
    # --------------------------------------------------------------
    def _load_experts(self):
        """Instantiate each base model and load its checkpoint."""
        experts = []
        for ckpt, arch in zip(self.ckpt_paths, self.arch_names):
            assert arch in _MODEL_MAP, f"Unknown architecture: {arch}"
            builder = _MODEL_MAP[arch]

            model = builder()
            state = torch.load(Path(ckpt), map_location=self.device)
            model.load_state_dict(state)
            model.eval().to(self.device)

            experts.append(model)
        return experts

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            The input batch (B, C, H, W).

        Returns
        -------
        Tensor
            Probabilities (B, num_classes).
        """
        logits = [m(x) for m in self.experts]        # list[(B,C)]
        avg_logits = torch.stack(logits, dim=0).mean(0)
        return F.softmax(avg_logits, dim=1)