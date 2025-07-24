# ensembles/weighted_voting.py
"""
Weighted-Voting Ensemble
~~~~~~~~~~~~~~~~~~~~~~~~
Same idea as Soft-Voting, but each expert contributes with a user-defined
positive weight.  The weights are normalised on the fly, so any scale
(e.g. [1, 1, 2]) is acceptable.
"""

from __future__ import annotations
from typing import List
import torch
import torch.nn.functional as F

from .soft_voting import SoftVotingEnsemble


class WeightedVotingEnsemble(SoftVotingEnsemble):
    """
    Parameters
    ----------
    ckpt_paths : list[str]
        Paths to N checkpoints.
    arch_names : list[str]
        Architecture string for each checkpoint (same length).
    weights : list[float]
        Relative weight per expert, > 0.
    device : str
        "cuda" or "cpu"
    """

    def __init__(
        self,
        ckpt_paths: List[str],
        arch_names: List[str],
        weights:    List[float],
        device: str = "cuda",
    ):
        assert len(weights) == len(ckpt_paths), "weights length mismatch"
        assert all(w > 0 for w in weights),     "weights must be positive"

        # store as tensor for fast ops later
        self.weights = torch.tensor(weights, dtype=torch.float32)

        # initialises self.experts via _load_experts() in parent
        super().__init__(ckpt_paths, arch_names, device)

    # ------------- forward pass -------------
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns class probabilities after weighted averaging of logits.
        """
        logits_list = [m(x) for m in self.experts]          # [(B,C)] * N
        w = self.weights.to(self.device).view(-1, 1, 1)     # (N,1,1)

        weighted_sum = (torch.stack(logits_list, 0) * w).sum(0)
        avg_logits   = weighted_sum / self.weights.sum()
        return F.softmax(avg_logits, dim=1)