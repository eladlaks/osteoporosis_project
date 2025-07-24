# ensembles/soft_voting.py
"""
Soft-Voting Ensemble – generic version.
"""
from .base import EnsembleModel
from ensembles.utils import build_model_from_ckpt     # ← כאן
import torch, torch.nn.functional as F

class SoftVotingEnsemble(EnsembleModel):
    """Average logits of N expert models."""

    def _load_experts(self):
        experts = []
        for ckpt, arch in zip(self.ckpt_paths, self.arch_names):
            model = build_model_from_ckpt(arch, ckpt, self.device)
            experts.append(model)
        return experts

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = [m(x) for m in self.experts]
        avg_logits = torch.stack(logits, 0).mean(0)
        return F.softmax(avg_logits, dim=1)