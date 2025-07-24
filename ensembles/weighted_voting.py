# ensemble/weighted_voting.py
"""
Weighted‐Voting Ensemble
~~~~~~~~~~~~~~~~~~~~~~~~
Like SoftVoting, but each expert can receive a user-supplied weight.
Weights are normalised internally so you may pass any positive scale.
"""

import torch
import torch.nn.functional as F

from .soft_voting import SoftVotingEnsemble


class WeightedVotingEnsemble(SoftVotingEnsemble):
    """
    Parameters
    ----------
    ckpt_paths : list[str]
        Checkpoints of the expert models.
    arch_names : list[str]
        Architecture identifiers, same length / same order as ckpt_paths.
    weights : list[float]
        Relative weight per expert (same length). Must all be > 0.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        ckpt_paths: list[str],
        arch_names: list[str],
        weights: list[float],
        device: str = "cuda",
    ):
        assert len(weights) == len(
            ckpt_paths
        ), "weights must match number of checkpoints"
        self.weights = torch.tensor(weights, dtype=torch.float32)
        super().__init__(ckpt_paths, arch_names, device)

    # only the forward changes – we reuse _load_experts from parent
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = [m(x) for m in self.experts]  # list[(B,C)]
        w = self.weights.to(self.device).view(-1, 1, 1)  # (M,1,1)
        weighted = torch.stack(logits, dim=0) * w
        avg_logits = weighted.sum(0) / self.weights.sum()
        return F.softmax(avg_logits, dim=1)