# ensembles/utils.py
"""
Utility helpers for ensemble loading.
"""

from pathlib import Path
import torch
import torch.nn as nn
from models.resnet_model import get_resnet_model
from models.vgg19_model  import get_vgg19_model
from models.vit_model    import get_vit_model
# add more imports if you train other backbones (dino, efficientnet …)

_MODEL_BUILDERS = {
    "resnet50": get_resnet_model,
    "vgg19"   : get_vgg19_model,
    "vit"     : get_vit_model,
    # "dinov2": get_dinov2_model,
}

def build_model_from_ckpt(arch: str, ckpt_path: str, device: str = "cpu"):
    """
    Build an architecture, adapt the FC-head to num_classes inferred
    from the checkpoint, load weights, return ready-to-eval model.
    """
    assert arch in _MODEL_BUILDERS, f"Unknown architecture {arch}"
    state = torch.load(Path(ckpt_path), map_location=device)

    # infer num_classes from last linear layer weight
    for k, v in state.items():
        if ".weight" in k and v.dim() == 2:        # Linear: (out,in)
            num_classes = v.size(0)               # out_features
            break
    else:
        raise RuntimeError("Could not infer num_classes from checkpoint")

    # build fresh model
    model = _MODEL_BUILDERS[arch](num_classes=num_classes)

    # if head shapes differ, ignore them while loading
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[{ckpt_path}] partial load – missing: {missing}  unexpected: {unexpected}")

    model.eval().to(device)
    return model