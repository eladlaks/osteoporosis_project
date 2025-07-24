# ensemble/stacking.py
"""
Stacking Ensemble
~~~~~~~~~~~~~~~~~
Feeds each expert's *probabilities* into a pre-trained meta-classifier
(e.g. LogisticRegression) and returns the meta-probabilities.
"""

import joblib, numpy as np, torch
from .base import EnsembleModel

# --- import your model builders here ---------------------------------
from models.resnet_model import get_resnet_model
from models.vgg19_model  import get_vgg19_model
from models.vit_model    import get_vit_model
# add others as needed
# ---------------------------------------------------------------------

_MODEL_MAP = {
    "resnet50": get_resnet_model,
    "vgg19":    get_vgg19_model,
    "vit":      get_vit_model,
}


class StackingEnsemble(EnsembleModel):
    """
    Parameters
    ----------
    ckpt_paths : list[str]
        Paths to expert checkpoints.
    arch_names : list[str]
        Architecture names (same order/length as ckpt_paths).
    meta_clf_path : str
        Pickled scikit-learn classifier trained on validation logits.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        ckpt_paths: list[str],
        arch_names: list[str],
        meta_clf_path: str,
        device: str = "cuda",
    ):
        self.meta = joblib.load(meta_clf_path)
        super().__init__(ckpt_paths, arch_names, device)

    # --------------------------------------------------------------
    # required abstract methods from EnsembleModel
    # --------------------------------------------------------------
    def _load_experts(self):
        models = []
        for ckpt, arch in zip(self.ckpt_paths, self.arch_names):
            assert arch in _MODEL_MAP, f"Unknown architecture: {arch}"
            model = _MODEL_MAP[arch](pretrained=False)
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.eval().to(self.device)
            models.append(model)
        return models

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns meta-probabilities as a torch.Tensor on the same device.
        """
        # collect base probabilities as NumPy
        probs = [
            torch.softmax(m(x), dim=1).cpu().numpy() for m in self.experts
        ]                                              # list[(B,C)]
        X = np.concatenate(probs, axis=1)             # (B, M·C)
        meta_probs = self.meta.predict_proba(X)       # (B, C) numpy
        return torch.from_numpy(meta_probs).to(self.device)